# Velocity-Constrained Hydronic (VCH) Sizing Framework
### First Edition | A Constraint-Driven Engineering Design Handbook

---

## 📌 Project Overview
The **Velocity-Constrained Hydronic (VCH) Sizing Framework** is an 11-step iterative design methodology engineered to bridge the gap between theoretical thermal calculations and physical fabrication constraints. 

Traditional hydronic sizing often treats thermal surface areas and fluid dynamics in isolation, leading to designs that are thermodynamically sound on paper but unworkable in practice. This framework introduces a unified workflow that simultaneously constraints fluid velocity, enforces standard ASME pipe schedules, and accounts for real-world coil packing efficiencies.

### Key Target Audience
* **Process & Mechanical Engineers** designing industrial thermal circulation loops.
* **Mechatronics & Automation Hobbyists** bridging physical plumbing with PLC control logic.
* **System Designers** looking for an auditable, step-by-step sizing protocol.

---

## 🛠️ Core Engineering Features
* **Independent Demand Separation:** Explicitly isolates Batch Processing and Balance Mode thermal loads prior to hardware allocation.
* **Velocity-Erosion Ceilings:** Enforces strict fluid velocity boundaries cross-referenced against **ASME B36.10M / B36.19M** nominal pipe sizes to prevent long-term internal pipe degradation.
* **Coil Packing Efficiency ($\eta_{pack}$):** Introduces a compound mathematical modifier accounting for real-world welding clearances, thermal expansion gaps, and fluid dead-zone prevention.
* **Automation-Ready Integration:** Features discrete-time control loops and state-machine maps designed for immediate PLC deployment.

---

## 📁 Repository Structure
```text
├── documentation/
│   └── VCH_Method_(FE).pdf  # The Core Design Handbook
├── Appendix/
│   └── Appendix.pdf          # Interactive calculation sheet
└── README.md                                   # Project landing page
