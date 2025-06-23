# Community Edition Licensing

The Modular Community Edition provides free access to MAX and Mojo for both personal and commercial use. This document explains the licensing terms in plain language.

## Key Points

### Free Forever
- The Community Edition is free and will remain free
- No hidden costs or surprise charges
- Full access to core features

### Commercial Use Allowed
- You can use MAX and Mojo for commercial products
- No revenue sharing or royalties required
- Simple notification requested for commercial use

### Hardware Limits
- **Unlimited**: CPUs and NVIDIA GPUs
- **Up to 8 devices**: Other accelerators (AMD GPUs, etc.)
- **Beyond 8 devices**: Contact Modular for Enterprise license

## License Summary

### What You Can Do ✅

1. **Personal Projects**
   - Any personal or hobby projects
   - Learning and experimentation
   - Open source contributions

2. **Commercial Use**
   - Production deployments
   - Commercial products and services
   - Internal business applications
   - Client projects

3. **Hardware Usage**
   - Unlimited CPU cores
   - Unlimited NVIDIA GPUs
   - Up to 8 AMD or other GPUs
   - Development on any machine

4. **Distribution**
   - Share your Mojo code
   - Distribute compiled binaries
   - Create libraries and frameworks
   - Build tools on top of MAX/Mojo

### What You Need to Do 📋

For commercial use, Modular asks (but doesn't require) that you:

1. **Notify them**: Email usage@modular.com
2. **Allow logo use**: Let them showcase your company as a user
3. **That's it!** No fees, no contracts

### What You Cannot Do ❌

1. **Remove attributions**: Keep Modular's copyright notices
2. **Misrepresent ownership**: Don't claim you created MAX/Mojo
3. **Exceed device limits**: Stay within 8 non-NVIDIA accelerators for commercial use

## Perpetual License Guarantee

The Community License includes an important protection:

> If Modular stops updating the software, you automatically get a perpetual license to continue using the last version.

This means:
- You won't lose access if Modular pivots or shuts down
- Your investment in MAX/Mojo code is protected
- More liberal than many proprietary licenses (including CUDA)

## Commercial Use Examples

### Allowed Under Community Edition

1. **SaaS Application**
   ```python
   # Running on 4 NVIDIA A100 GPUs - ✅ Allowed
   max serve --model-path=llama-70b --tensor-parallel-size=4
   ```

2. **On-Premise Deployment**
   ```python
   # Running on 100 CPU cores - ✅ Allowed
   max serve --model-path=model --device=cpu
   ```

3. **Edge Deployment**
   ```python
   # Running on 8 AMD GPUs - ✅ Allowed (at the limit)
   max serve --model-path=model --device=rocm
   ```

### Requires Enterprise License

1. **Large AMD GPU Cluster**
   ```python
   # Running on 16 AMD GPUs - ❌ Needs Enterprise
   # Contact sales@modular.com
   ```

## Notification Process

For commercial use, send a simple email to usage@modular.com:

```
Subject: Commercial Use Notification - [Your Company]

Hi Modular Team,

We're using MAX/Mojo in production at [Company Name].

Use case: [Brief description]
Website: [Your website]

We're happy to be featured as a user.

Best,
[Your name]
```

## FAQ

### Q: Do I need to pay for commercial use?
**A**: No, commercial use is free under the Community Edition.

### Q: What if I'm a consultant using MAX for client work?
**A**: That's allowed! It counts as commercial use, so please notify Modular.

### Q: Can I use MAX in my startup's product?
**A**: Yes! The Community Edition is perfect for startups.

### Q: What about internal company tools?
**A**: Also allowed under commercial use provisions.

### Q: Do I need a license for development?
**A**: No, development and testing are always free.

### Q: Can I benchmark MAX against other frameworks?
**A**: Yes, benchmarking and comparisons are allowed.

### Q: What if I exceed 8 non-NVIDIA GPUs accidentally?
**A**: Contact Modular to discuss Enterprise options. They're reasonable!

## Open Source Projects

Using MAX/Mojo in open source projects is encouraged:

- No notification required for open source
- Can use unlimited hardware
- Consider adding a "Powered by MAX" badge
- Share your projects with the community

## License Text

The full legal license is available at:
https://www.modular.com/legal/community

Key sections:
- Grant of rights (Section 2)
- Commercial use (Section 3)
- Hardware limitations (Section 4)
- Perpetual license clause (Section 7)

## Comparison with Other Licenses

| Aspect | Modular Community | CUDA | PyTorch | TensorFlow |
|--------|------------------|------|---------|------------|
| Commercial Use | ✅ Free | ✅ Free* | ✅ Free | ✅ Free |
| Source Available | ❌ | ❌ | ✅ | ✅ |
| Perpetual Clause | ✅ | ❌ | N/A | N/A |
| Hardware Limits | Some | None | None | None |
| Attribution | Required | Required | Required | Required |

*CUDA has complex licensing for certain use cases

## Enterprise Benefits

If you need more than the Community Edition offers:

1. **Unlimited Hardware**: No device restrictions
2. **Priority Support**: Direct access to Modular team
3. **Custom Features**: Influence roadmap
4. **SLAs**: Guaranteed response times
5. **Training**: Onboarding and optimization help

Contact: sales@modular.com

## Summary

The Modular Community License is designed to be:
- **Developer-friendly**: Use it for anything
- **Startup-friendly**: No costs as you grow
- **Future-proof**: Perpetual license protection
- **Simple**: No complex terms or restrictions

For most users and companies, the Community Edition provides everything needed to build and deploy AI applications with MAX and Mojo.