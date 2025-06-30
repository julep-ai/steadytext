# D1 Cache Proxy Worker

This Cloudflare Worker provides a REST API proxy for SteadyText's D1 cache backend.

## Setup

### Prerequisites

1. A Cloudflare account with Workers enabled
2. Wrangler CLI installed: `npm install -g wrangler`
3. Node.js 16.17.0 or later

### Installation

1. Install dependencies:
   ```bash
   npm install
   ```

2. Create a D1 database:
   ```bash
   npx wrangler d1 create steadytext-cache
   ```

3. Update `wrangler.toml` with your database ID from the output above.

4. Initialize the database schema:
   ```bash
   npx wrangler d1 execute steadytext-cache --file=src/schema.sql
   ```

### Configuration

1. Generate an API key for authentication:
   ```bash
   openssl rand -base64 32
   ```

2. Set the API key as a secret:
   ```bash
   npx wrangler secret put API_KEY
   ```

### Development

Run the Worker locally:
```bash
npm run dev
```

### Deployment

Deploy to Cloudflare Workers:
```bash
npm run deploy
```

The deployment will provide you with a URL like:
`https://d1-cache-proxy.<your-subdomain>.workers.dev`

## Usage with SteadyText

Configure SteadyText to use the D1 backend:

```bash
export STEADYTEXT_CACHE_BACKEND=d1
export STEADYTEXT_D1_API_URL=https://d1-cache-proxy.<your-subdomain>.workers.dev
export STEADYTEXT_D1_API_KEY=<your-api-key>
```

Or in Python:
```python
import os
os.environ["STEADYTEXT_CACHE_BACKEND"] = "d1"
os.environ["STEADYTEXT_D1_API_URL"] = "https://d1-cache-proxy.<your-subdomain>.workers.dev"
os.environ["STEADYTEXT_D1_API_KEY"] = "<your-api-key>"

from steadytext import generate
text = generate("Hello world")  # Uses D1 cache
```

## API Endpoints

All endpoints require Bearer token authentication.

- `POST /api/init` - Initialize a cache table
- `POST /api/get` - Get a single value
- `POST /api/set` - Set a single value
- `POST /api/batch/get` - Get multiple values
- `POST /api/batch/set` - Set multiple values
- `POST /api/clear` - Clear all cache entries
- `POST /api/stats` - Get cache statistics

## Monitoring

View real-time logs:
```bash
npm run tail
```

## Performance Considerations

1. **Batch Operations**: Use batch get/set operations when possible to reduce latency
2. **Regional Deployment**: Deploy to regions close to your users
3. **Cache Size**: Monitor cache size and adjust `max_size_mb` as needed
4. **Rate Limits**: Be aware of Cloudflare's rate limits for Workers and D1

## Security

1. Always use HTTPS
2. Keep your API key secret
3. Consider adding IP allowlists or additional authentication
4. Monitor access logs for suspicious activity

## Troubleshooting

### Database not found
Ensure the database ID in `wrangler.toml` matches the output from `wrangler d1 create`

### Authentication errors
Verify the API_KEY secret is set correctly: `npx wrangler secret list`

### Performance issues
- Check D1 query performance in the Cloudflare dashboard
- Consider increasing Worker CPU limits if needed
- Use batch operations for multiple cache operations