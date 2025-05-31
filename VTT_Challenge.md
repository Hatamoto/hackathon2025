They're essentially aiming to **deduplicate and connect innovation disclosures** that may be described differently across various public sources, with a special focus on **VTT’s collaborations**. Here's a breakdown:

---

### 🔍 **Goal Overview:**

To **map VTT’s innovation partnerships** based on **public disclosures from other organizations**, and to:

1. **Identify when different disclosures refer to the same innovation** (even if described differently).
    
2. **Aggregate and structure this data** to represent each unique innovation clearly—without losing track of where each piece of information came from.
    

---

### 🧩 **Challenge Step 1 – "Innovative Resolution":**

**Main Question:**

> When are different sources discussing the same innovation?

**Your Task:**  
Detect **duplicates** or **overlapping innovations** across various disclosures (websites, press releases, etc.), even if phrased differently.

**How:**

- Use cues like:
    
    - **Names of associated organizations**
        
    - **Descriptions**
        
    - **Raw text**
        
- Consider NLP methods for:
    
    - Semantic similarity
        
    - Entity matching
        
    - Fuzzy deduplication
        

**Why:**  
Different organizations might mention the same joint innovation using different words—your job is to link those.

---

### 🔗 **Challenge Step 2 – "Innovation Relationship":**

**Main Question:**

> How are the different disclosures related?

**Your Task:**  
Create a **consolidated representation** of each innovation that includes:

- All the relevant info (e.g. who worked on it, what it does)
    
- All the **original sources** and their version of the description
    

**How:**

- Design a data structure that:
    
    - Aggregates info across sources
        
    - Keeps traceability (source IDs, URLs, timestamps, etc.)
        

**Why:**  
This enables downstream tasks like:

- Innovation tracking over time
    
- Analyzing VTT’s role in collaborative R&D
    
- Avoiding duplication in innovation impact metrics
    

---

### 🧠 What They’re Testing:

- Your ability to do **semantic deduplication**
    
- Your data modeling skill to **capture relationships and preserve provenance**
    
- Probably your **NLP and knowledge graph instincts** too