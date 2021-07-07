TVM+TIDL: TVM with TIDL Offload
===============================

We leverage TVM's BYOC (Bring Your Own Codegen) mechanism to offload subgraphs to
TIDL (Texas Instruments' Deep Learning library) for accelerated execution on TI's
J7 family of SoCs.  Layers unsupported by TIDL are left with TVM code generation
and runtime.

Branches
--------
  * tidl-j7 - This is the release branch
  * add-tidl-develop - This is the internal development branch
    (in sync with TIDL development branch)

Tags
----
  TVM+TIDL releases are synchronized and packaged in TI's PSDK (Processor SDK) releases.
The following are the TVM+TIDL tags for PSDK releases.

|  PSDK release  |  TVM+TIDL release tag  |  TVM+TIDL other tags                                  |
|----------------|------------------------|-------------------------------------------------------|
| 8.0            | TIDL\_PSDK\_8.0        |                                                       |
| 7.3            | TIDL\_PSDK\_7.3        | TIDL\_PSDK\_7.3\_UPDATE1                              |

  Suffix "RC" stands for release candidates, suffix "UPDATE" stands for updates
that are still compatible with certain releases.

