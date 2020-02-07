gcc -c -fPIC -DPLATFORM_64BIT tidl_relayImport.c -o tidl_relayImport.o
gcc -c -fPIC -DPLATFORM_64BIT tidl_import_utils.c -o tidl_import_utils.o
gcc -shared -o tidl_relayImport.so ./tidl_relayImport.o ./tidl_import_utils.o
