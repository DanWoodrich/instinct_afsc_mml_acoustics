This is a repository for a contributor module for DanWoodrich/INSTINCT 

Currently, as the only active contributor module to INSTINCT, this project will be a necessary template to understand how to 
design other contributor modules. For this reason, more extensive documentation on this module will follow shortly. 

To populate your INSTINCT user folder with this module, first run in git bash

```bash
git submodule update --remote --merge
```

And then run in cmd

```bash
instinct pull_contrib instinct_afsc_mml_acoustics
```

To run the pipelines from this project out of the box, note that at minimum you'll have to edit the files in data which reference 
annotations and sound files (in ./Data), and parameters files (in ./Projects). 

Parameters files and pipelines use [nestedtext](https://github.com/KenKundert/nestedtext), which is a standard for INSTINCT. 