import { Hono } from "hono";
import { bearerAuth } from "hono/bearer-auth";
import { logger } from "hono/logger";
import { prettyJSON } from "hono/pretty-json";

type Bindings = {
  DB: D1Database;
  API_KEY: string;
};

type CacheEntry = {
  key: string;
  value: string;
  frequency: number;
  last_access: number;
  size: number;
  created_at: number;
};

const app = new Hono<{ Bindings: Bindings }>();

// Middleware setup
app.use("*", prettyJSON(), logger(), async (c, next) => {
  const auth = bearerAuth({ token: c.env.API_KEY });
  return auth(c, next);
});

// Helper function to get current timestamp in microseconds
function getCurrentTimeMicros(): number {
  return Date.now() * 1000;
}

// Helper function to ensure cache table exists
async function ensureCacheTable(db: D1Database, cacheName: string): Promise<void> {
  const tableName = `${cacheName}_cache`;
  
  await db.prepare(`
    CREATE TABLE IF NOT EXISTS ${tableName} (
      key TEXT PRIMARY KEY,
      value TEXT NOT NULL,
      frequency INTEGER DEFAULT 1,
      last_access INTEGER NOT NULL,
      size INTEGER NOT NULL,
      created_at INTEGER NOT NULL
    )
  `).run();
  
  await db.prepare(`
    CREATE INDEX IF NOT EXISTS idx_${cacheName}_frecency 
    ON ${tableName}(frequency DESC, last_access DESC)
  `).run();
}

// Helper function to calculate frecency score
function calculateFrecencyScore(frequency: number, lastAccess: number, currentTime: number): number {
  const timeSinceAccess = (currentTime - lastAccess) / 1000000; // Convert to seconds
  return frequency / (1 + timeSinceAccess);
}

// Initialize cache table
app.post("/api/init", async (c) => {
  try {
    const { cache_name, max_size_mb } = await c.req.json();
    
    await ensureCacheTable(c.env.DB, cache_name);
    
    // Store metadata
    const maxSizeBytes = Math.floor(max_size_mb * 1024 * 1024);
    const now = getCurrentTimeMicros();
    
    await c.env.DB.prepare(`
      INSERT OR REPLACE INTO cache_metadata (cache_name, max_size_bytes, created_at, last_updated)
      VALUES (?, ?, ?, ?)
    `).bind(cache_name, maxSizeBytes, now, now).run();
    
    return c.json({ success: true, message: "Cache initialized" });
  } catch (error) {
    return c.json({ error: `Failed to initialize cache: ${error}` }, 500);
  }
});

// Get single value
app.post("/api/get", async (c) => {
  try {
    const { cache_name, key } = await c.req.json();
    const tableName = `${cache_name}_cache`;
    
    // Get entry and update frecency
    const result = await c.env.DB.prepare(`
      SELECT value, frequency FROM ${tableName} WHERE key = ?
    `).bind(key).first<CacheEntry>();
    
    if (!result) {
      return c.json({ found: false });
    }
    
    // Update frecency metadata
    const now = getCurrentTimeMicros();
    await c.env.DB.prepare(`
      UPDATE ${tableName} 
      SET frequency = frequency + 1, last_access = ?
      WHERE key = ?
    `).bind(now, key).run();
    
    return c.json({ found: true, value: result.value });
  } catch (error) {
    return c.json({ error: `Failed to get value: ${error}` }, 500);
  }
});

// Set single value
app.post("/api/set", async (c) => {
  try {
    const { cache_name, key, value, size } = await c.req.json();
    const tableName = `${cache_name}_cache`;
    const now = getCurrentTimeMicros();
    
    // Check if we need to evict entries
    const metadata = await c.env.DB.prepare(`
      SELECT max_size_bytes FROM cache_metadata WHERE cache_name = ?
    `).bind(cache_name).first<{ max_size_bytes: number }>();
    
    if (metadata) {
      const currentSize = await c.env.DB.prepare(`
        SELECT COALESCE(SUM(size), 0) as total_size FROM ${tableName}
      `).first<{ total_size: number }>();
      
      if (currentSize && currentSize.total_size + size > metadata.max_size_bytes) {
        // Evict entries based on frecency
        await evictEntries(c.env.DB, tableName, metadata.max_size_bytes, size);
      }
    }
    
    // Insert or update entry
    await c.env.DB.prepare(`
      INSERT OR REPLACE INTO ${tableName} (key, value, frequency, last_access, size, created_at)
      VALUES (?, ?, 
        COALESCE((SELECT frequency + 1 FROM ${tableName} WHERE key = ?), 1),
        ?, ?, ?)
    `).bind(key, value, key, now, size, now).run();
    
    return c.json({ success: true });
  } catch (error) {
    return c.json({ error: `Failed to set value: ${error}` }, 500);
  }
});

// Batch get
app.post("/api/batch/get", async (c) => {
  try {
    const { cache_name, keys } = await c.req.json();
    const tableName = `${cache_name}_cache`;
    const now = getCurrentTimeMicros();
    
    const results: Record<string, any> = {};
    
    // Process in chunks to avoid query size limits
    const chunkSize = 50;
    for (let i = 0; i < keys.length; i += chunkSize) {
      const chunk = keys.slice(i, i + chunkSize);
      const placeholders = chunk.map(() => "?").join(",");
      
      const entries = await c.env.DB.prepare(`
        SELECT key, value, frequency FROM ${tableName} 
        WHERE key IN (${placeholders})
      `).bind(...chunk).all<CacheEntry>();
      
      // Update frecency for found entries
      if (entries.results.length > 0) {
        const updatePromises = entries.results.map(entry => 
          c.env.DB.prepare(`
            UPDATE ${tableName} 
            SET frequency = frequency + 1, last_access = ?
            WHERE key = ?
          `).bind(now, entry.key).run()
        );
        await Promise.all(updatePromises);
      }
      
      // Build results
      for (const entry of entries.results) {
        results[entry.key] = { found: true, value: entry.value };
      }
    }
    
    // Mark missing keys
    for (const key of keys) {
      if (!(key in results)) {
        results[key] = { found: false };
      }
    }
    
    return c.json({ results });
  } catch (error) {
    return c.json({ error: `Failed to batch get: ${error}` }, 500);
  }
});

// Batch set
app.post("/api/batch/set", async (c) => {
  try {
    const { cache_name, items } = await c.req.json();
    const tableName = `${cache_name}_cache`;
    const now = getCurrentTimeMicros();
    
    // Calculate total size of new items
    const totalNewSize = items.reduce((sum: number, item: any) => sum + item.size, 0);
    
    // Check if we need to evict
    const metadata = await c.env.DB.prepare(`
      SELECT max_size_bytes FROM cache_metadata WHERE cache_name = ?
    `).bind(cache_name).first<{ max_size_bytes: number }>();
    
    if (metadata) {
      const currentSize = await c.env.DB.prepare(`
        SELECT COALESCE(SUM(size), 0) as total_size FROM ${tableName}
      `).first<{ total_size: number }>();
      
      if (currentSize && currentSize.total_size + totalNewSize > metadata.max_size_bytes) {
        await evictEntries(c.env.DB, tableName, metadata.max_size_bytes, totalNewSize);
      }
    }
    
    // Insert all items in a transaction
    const statements = items.map((item: any) =>
      c.env.DB.prepare(`
        INSERT OR REPLACE INTO ${tableName} (key, value, frequency, last_access, size, created_at)
        VALUES (?, ?, 
          COALESCE((SELECT frequency + 1 FROM ${tableName} WHERE key = ?), 1),
          ?, ?, ?)
      `).bind(item.key, item.value, item.key, now, item.size, now)
    );
    
    await c.env.DB.batch(statements);
    
    return c.json({ success: true });
  } catch (error) {
    return c.json({ error: `Failed to batch set: ${error}` }, 500);
  }
});

// Clear cache
app.post("/api/clear", async (c) => {
  try {
    const { cache_name } = await c.req.json();
    const tableName = `${cache_name}_cache`;
    
    await c.env.DB.prepare(`DELETE FROM ${tableName}`).run();
    
    return c.json({ success: true });
  } catch (error) {
    return c.json({ error: `Failed to clear cache: ${error}` }, 500);
  }
});

// Get statistics
app.post("/api/stats", async (c) => {
  try {
    const { cache_name } = await c.req.json();
    const tableName = `${cache_name}_cache`;
    
    const stats = await c.env.DB.prepare(`
      SELECT 
        COUNT(*) as entry_count,
        COALESCE(SUM(size), 0) as total_size,
        COALESCE(AVG(frequency), 0) as avg_frequency,
        COALESCE(MAX(frequency), 0) as max_frequency
      FROM ${tableName}
    `).first();
    
    const metadata = await c.env.DB.prepare(`
      SELECT max_size_bytes FROM cache_metadata WHERE cache_name = ?
    `).bind(cache_name).first<{ max_size_bytes: number }>();
    
    return c.json({
      ...stats,
      max_size_bytes: metadata?.max_size_bytes || 0,
      utilization: metadata && stats ? (stats.total_size / metadata.max_size_bytes) : 0,
    });
  } catch (error) {
    return c.json({ error: `Failed to get stats: ${error}` }, 500);
  }
});

// Helper function to evict entries based on frecency
async function evictEntries(
  db: D1Database,
  tableName: string,
  maxSizeBytes: number,
  sizeToAdd: number
): Promise<void> {
  const targetSize = Math.floor(maxSizeBytes * 0.8); // Target 80% capacity
  const currentTime = getCurrentTimeMicros();
  
  // Get entries sorted by frecency score (lowest first)
  const entries = await db.prepare(`
    SELECT key, size,
           frequency * 1.0 / (1 + (? - last_access) / 1000000.0) as frecency_score
    FROM ${tableName}
    ORDER BY frecency_score ASC
  `).bind(currentTime).all<{ key: string; size: number }>();
  
  let totalSize = entries.results.reduce((sum, entry) => sum + entry.size, 0);
  const keysToDelete: string[] = [];
  
  // Remove entries until we have enough space
  for (const entry of entries.results) {
    if (totalSize - entry.size + sizeToAdd <= targetSize) {
      break;
    }
    keysToDelete.push(entry.key);
    totalSize -= entry.size;
  }
  
  // Delete selected entries
  if (keysToDelete.length > 0) {
    const placeholders = keysToDelete.map(() => "?").join(",");
    await db.prepare(`
      DELETE FROM ${tableName} WHERE key IN (${placeholders})
    `).bind(...keysToDelete).run();
  }
}

export default app;