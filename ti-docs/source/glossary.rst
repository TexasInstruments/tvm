.. _ti-tvm-glossary:

#############################
Glossary
#############################

.. glossary::

   Apache TVM
      An open source machine learning compiler framework for CPUs, GPUs, and other devices. It enables running and optimization of machine learning computations on such hardware.

   compute
      A mathematical formula that specifies what an operator does.

   DLR
      Deep Learning Runtime from Amazon AWS. For the purpose of running TVM compiled models with the DLR runtime, DLR is simply a wrapper around the TVM runtime.


   inference
      Inferencing is the act of using a trained deep learning network to output a prediction about incoming data.

   MMA
      The Matrix Multiplication Accelerator (MMA) is a key hardware accelerator on TDA4 processors. The MMA provides highly parallel deep learning instructions. It is architected to optimize data flow management for deep learning, while minimizing power and external memory devices. The MMA is accessed as an extension of the C71x instruction set, and leverages the same highly parallel data path as the C71x. 

   Relay IR
      TVM’s internal common representation for machine learning models.

   schedule
      Learning rate schedules allow you to optimize the training of a machine learning network. Such optimization is possible because increased performance and faster training can be achieved using a learning rate that changes during training. Common learning rate schedules include time-based decay, step decay, and exponential decay.

   subgraphs
      In machine learning, a graph is a data structure that consists of nodes (also called vertices) and edges that connect these nodes. A subgraph is a subset of these nodes and edges.  Subgraphs can be useful in machine learning for purposes such as identifying patterns or relationships within the data or for simplifying a complex graph.

   TDA4 family
      The TDA4VM family includes dual Arm Cortex-A72 SoC and C7x DSPs. It is designed for applications in deep-learning, vision and multimedia.

   TIDL
      TI Deep Learning library is TI's software ecosystem for deep learning algorithm (CNN) acceleration. It contains highly optimized implementations of common layers on C7x/MMA. TI TVM can offload some of its computations to TIDL.

   TVM
      Tensor Virtual Machine (TVM) is a compiler stack used to compile various deep learning models from different frameworks to specialized CPU, GPU or other accelerator architectures. 

