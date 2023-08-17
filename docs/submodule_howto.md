# How to work with submodules
Packages imported from other Gitlab repositories are called submodules. They correspond to a reference to a specific commit.  


To install a repository containing one or multiple submodules:
```
$ git clone --recurse-submodules [path_to_repository] 
```
or if you already have the repository cloned:
```
$ git submodule update --init --recursive
```

If the submodule has been updated on its repository, it needs to be updated here. There are two methods:
```
$ cd [submodule_folder]
$ git fetch
$ git pull origin [branch]
```
or
```
$ git submodule update --remote
```

To check changes in your submodules when doing a `$ git status` command, you can change you configuration with:
```
$ git config status.submodulesummary 1
```