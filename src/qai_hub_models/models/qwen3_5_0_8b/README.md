> [!WARNING]
> This model is not published. Use with caution; it may not meet performance/accuracy standards and may not support some runtimes or chipsets/devices. We do not provide support for unpublished models. If this model was previously published, use earlier releases.

# [Qwen3.5-0.8B: Ultra-compact 0.8B parameter model from the Qwen3.5 series for edge deployment](https://aihub.qualcomm.com/models/qwen3_5_0_8b)

Qwen3.5 is the latest multilingual language model series from Alibaba Cloud with improved reasoning and instruction-following capabilities over Qwen3.

This is based on the implementation of Qwen3.5-0.8B found [here](https://huggingface.co/Qwen/Qwen3.5-0.8B).
This repository contains scripts for optimized on-device export suitable to run on Qualcomm® devices. More details on model performance across various devices, can be found [here](https://aihub.qualcomm.com/models/qwen3_5_0_8b).

Qualcomm AI Hub Models uses [Qualcomm AI Hub Workbench](https://workbench.aihub.qualcomm.com) to compile, profile, and evaluate this model. [Sign up](https://myaccount.qualcomm.com/signup) to run these models on a hosted Qualcomm® device.

## Quick Start

Use our lightweight command-line interface to inspect and download Qwen3.5-0.8B:

```bash
pip install qai_hub_models_cli # (the CLI is also available with the qai-hub-models package)

# Inspect the model and list the available download options
qai-hub-models info Qwen3.5-0.8B

# Print performance and accuracy metrics
qai-hub-models perf Qwen3.5-0.8B
qai-hub-models numerics Qwen3.5-0.8B

# Download a ready-to-deploy asset
qai-hub-models fetch Qwen3.5-0.8B --runtime qnn_context_binary --precision q4_0
```
See the [CLI README](../../../../cli/README.md)
for the full list of commands and filters.

## License
* The license for the original implementation of Qwen3.5-0.8B can be found
  [here](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/LICENSE).

## References
* [Qwen3.5: Towards Native Multimodal Agents](https://qwen.ai/blog?id=qwen3.5)
* [Source Model Implementation](https://huggingface.co/Qwen/Qwen3.5-0.8B)

## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).

## Usage and Limitations

This model may not be used for or in connection with any of the following applications:

- Accessing essential private and public services and benefits;
- Administration of justice and democratic processes;
- Assessing or recognizing the emotional state of a person;
- Biometric and biometrics-based systems, including categorization of persons based on sensitive characteristics;
- Education and vocational training;
- Employment and workers management;
- Exploitation of the vulnerabilities of persons resulting in harmful behavior;
- General purpose social scoring;
- Law enforcement;
- Management and operation of critical infrastructure;
- Migration, asylum and border control management;
- Predictive policing;
- Real-time remote biometric identification in public spaces;
- Recommender systems of social media platforms;
- Scraping of facial images (from the internet or otherwise); and/or
- Subliminal manipulation
