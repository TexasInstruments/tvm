These files are needed for tests on AM57 target.
First, create emulation of shared library with ./mk.sh (this needs to be executed on target).
Generated libtidl.so has placeholder implementation of TidlRunSubgraph(), that only prints (to stdout) input and output arguments.
Also, we need to copy deploy_param.params, deploy_graph.json and deploy_lib.tar, to current folder.
Then test can be executed: python3 ./test_target_tidl.py

