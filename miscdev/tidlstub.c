/** gcc -shared -o libfoo.so foo.o **/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void TidlRunSubgraph(int total_subgraphs, 
                     int subgraph_id, 
                     int num_inputs, 
                     int num_outputs, float **inputTensors, float **outputTensors)
{
   int i;
   float *data_in  = inputTensors[0];
   float *data_out = outputTensors[0];
   printf("DJDBG...inside TidlRunSubgraph, total_subgraphs=%d\n", total_subgraphs);
   printf("DJDBG...inside TidlRunSubgraph, subgraph_id =%d\n", subgraph_id);
   printf("DJDBG...inside TidlRunSubgraph, num_inputs  =%d\n", num_inputs);
   printf("DJDBG...inside TidlRunSubgraph, num_outputs =%d\n", num_outputs);
   printf("\nDJDBG...input values (first 5 values):\n");
   for (i = 0; i < 5; i ++) printf("%f ", *data_in++);
   printf("\nDJDBG...output values (first 3 values):\n");
   for (i = 0; i < 3; i ++)  printf("%f ", *data_out++);
   printf("\n-----\n");
}


