# Configuration Management

Work Journal allows you to create multiple named configurations and switch between them.

## What are Configurations?

A **configuration** is a named set of model assignments for the three functions:
- **Conversation**: Interactive chat and entry refinement
- **Processing**: Structures raw entries into professional summaries  
- **JIRA Matching**: Finds relevant tickets for your work

You can create multiple configurations and switch between them as needed.

## Managing Configurations

### View Current Configuration
The active configuration is shown in System menu → Model configuration.

### Switch Configurations
1. Access **System menu** (`x`)
2. Choose **Model configuration** (`1`)
3. Select numbered configuration from list
4. Configuration switches immediately

### Create New Configuration
1. Access **System menu** (`x`)
2. Choose **Model configuration** (`1`)  
3. Select **Create New Configuration** (`1`)
4. Follow the 4-step guided process

### Test Configuration
Use **Test Current Setup** (`t`) in the model configuration menu to verify all models work properly.

## Configuration Storage

Configurations are stored locally in `~/.work-journal/settings.json` and can be backed up and restored.

## Troubleshooting

**Configuration Won't Switch**: Ensure all models in the target configuration are available and API keys are set.

**Missing Models**: Check that local services are running or cloud API keys are configured.

The built-in help system (`h` key) provides detailed guidance for all configuration operations!