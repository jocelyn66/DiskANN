```mermaid
---
config:
  theme: default
---
graph TD  
    A[开始] --> B{是否有咖啡?}  
    B -->|是| C[喝咖啡]  
    B -->|否| D[买咖啡111]  
    C --> E[工作]  
    D --> E  
    E --> F[结束]
    
```