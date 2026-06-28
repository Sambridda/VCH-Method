# Velocity-Constrained Hydronic (VCH) Sizing Framework
### Second Edition
#### An Engineering Design Methodology for Industrial Hydronic Thermal Systems

---

## Overview

The **Velocity-Constrained Hydronic (VCH) Sizing Framework (VCH)** is a comprehensive engineering methodology for the thermal, hydraulic, and control-system design of submerged hydronic heating and cooling systems used in industrial liquid processing applications.

The framework was developed to address a fundamental limitation in conventional hydronic design practices, where thermal sizing, hydraulic analysis, equipment selection, and control strategy are frequently performed as independent calculations. Such fragmented workflows often produce systems that satisfy theoretical performance requirements while introducing hydraulic instability, fabrication constraints, or long-term operational deficiencies during implementation.

The VCH methodology integrates these disciplines into a unified **eleven-step iterative engineering workflow** progressing from thermal demand definition through hydraulic design, heat transfer analysis, environmental heat-loss modelling, coil optimisation, thermal driver selection, thermal battery design, control architecture, and formal engineering verification.

Rather than selecting hardware first and adapting the design around available components, the framework establishes independent thermal demand targets before equipment selection, ensuring that every subsequent engineering decision is driven by quantified system requirements.

---

## Scope

The methodology is intended for the design of industrial hydronic thermal regulation systems, including:

- Bulk liquid heating and cooling systems
- Process storage vessels
- Industrial melting and holding tanks
- Pharmaceutical and food processing facilities
- Renewable-assisted thermal energy systems
- Closed-loop thermal circulation systems

Although developed around submerged hydronic coil systems, the underlying methodology is applicable to a broad range of industrial process heating and thermal management applications.

---

## Principal Features

### Independent Thermal Demand Definition

Separates **Batch Processing Target** and **Balance Target** thermal demands before hardware selection, eliminating circular design dependencies and establishing a fixed engineering objective throughout the design process.

### Velocity-Constrained Hydraulic Design

Enforces fluid velocity limits using commercially available **ASME B36.10M** and **ASME B36.19M** Nominal Pipe Sizes, reducing long-term erosion, hydraulic instability, and unnecessary pressure losses.

### Fabrication-Aware Coil Optimisation

Introduces a compound coil packing efficiency correction accounting for:

- Structural support requirements
- Weld accessibility
- Thermal expansion clearances
- Manufacturing tolerances
- Hydraulic dead-zone prevention

### Integrated Heat Transfer Analysis

Employs the **Effectiveness–NTU** method with independent steady-state and cold-start heat transfer coefficient assessments for accurate evaluation of thermal performance.

### Environmental Heat-Loss Modelling

Implements a multilayer series thermal resistance network referenced to conservative worst-case ambient conditions for realistic prediction of standby and operational heat losses.

### Thermal Battery Engineering

Provides a complete methodology for sizing dedicated thermal storage systems, including vessel geometry, hydraulic priming constraints, thermal driver stability, and operational cycling behaviour.

### Industrial Control Integration

Extends beyond mechanical sizing through:

- PLC implementation philosophy
- IEC 61131-3 compliant control architecture
- Distributed control strategies
- PID controller derivations
- Instrumentation and sensor placement guidelines

### Formal Engineering Verification

Concludes with a structured verification stage confirming thermal, hydraulic, mechanical, and operational compliance before procurement or construction.

---

## Intended Audience

This handbook is intended for:

- Mechanical Engineers
- Process Engineers
- Mechatronics Engineers
- Automation & Controls Engineers
- Industrial System Designers
- Engineering students interested in industrial thermal system design

---

## Engineering Philosophy

The VCH Sizing Framework adopts a **constraint-driven engineering philosophy** in which thermal performance, hydraulic behaviour, manufacturability, and control implementation are evaluated simultaneously rather than sequentially.

By integrating first-principles thermodynamics, fluid mechanics, heat transfer, industrial automation, and practical fabrication constraints into a single iterative workflow, the methodology provides an auditable engineering process capable of producing physically grounded and industrially deployable hydronic thermal systems.

---

## Publication Information

**Title:** Velocity-Constrained Hydronic (VCH) Sizing Framework  
**Edition:** Second Edition  
**Author:** Sambridda Ranabhat  
**License:** CC BY-NC 4.0

---

## Citation

If you use or reference this methodology in academic or industrial work, please cite the published handbook accordingly.
