# Compatibility Models
The description of these models will be published in the proceedings of NAACL 2015 under
the title "[So similar and yet incompatible: Toward the automated identification of semantically compatible words](http://clic.cimec.unitn.it/marco/publications/kruszewski-baroni-compatibility-naacl-2015.pdf)"

## Requirements
[Torch7](http://torch.ch/)

## Usage

1. Go to the directory model/torch and edit my_config.lua to point to the root of your local copy of this repository

2. Create a "run configuration" (see the run_configs directory for examples)

3. Train the model:
```bash
luajit do-train.lua run_configs/conf_01.lua
```

4. Use the model to generate predictions:
```bash
luajit do-measure.lua run_configs/conf_01.lua
```

A pretrained configuration with the l2_interaction model is available at l2_interaction_config.lua

## Troubleshooting

### When I run the script, I get "luajit: do-train.lua:6: module 'models' not found:"

You need to add "./?/init.lua" to your LUA_PATH. One simple way of doing so is copy-pasting this command in your terminal:

    `echo "export LUA_PATH=$(luajit -e 'print(package.path)');./?/init.lua"`

