# CYOA (Choose Your Own Anomaly) Architecture Pipeline

This document maps exactly how the data flows from the physical telescope all the way to your Active Learning loop inside SkyPortal.

## The End-To-End Machine Learning System

```mermaid
graph TD

    %% Stage 1: The Firehose
    subgraph S1 [1. The Global Stream]
        ZTF[ZTF Telescope] --> |Millions of Alerts| Delta[Delta Supercomputer]
        Delta --> |AppleCiDEr Rust Engine| Base[Frozen 124-dim Vectors]
        Base --> |Physics features| XGBoost[XGBoost Classifier]
    end

    %% Stage 2: The CYOA Triage (What we built)
    subgraph S2 [2. The Anomaly Filter Local]
        Base --> |Vectors| IF[Isolation Forest Filter]
        XGBoost --> |Low Confidence| IF
        IF --> |Flags 20 targets| Profile[Profile Classifiers e.g. SN Hunter]
        
        Profile --> |Ranking| LLM[Llama-3.3 LLM Triage]
        LLM --> |cyoa_reasoning| Verdict[triage_verdicts.jsonl]
    end

    %% Stage 3: The SkyPortal Push
    subgraph S3 [3. SkyPortal Backend Integration]
        Verdict --> |push_to_skyportal.py| BoomAPI[/api/boom/alerts/]
        Verdict --> |push_to_skyportal.py| AnnotAPI[/api/sources/oid/annotations]
        BoomAPI --> FritzDB[(Fritz Database)]
        AnnotAPI --> FritzDB
    end

    %% Stage 4: Astronomer Interface
    subgraph S4 [4. SkyPortal Frontend / React]
        FritzDB --> UI[CYOAWidget.jsx]
        UI --> |Displays reasoning| Astro((Working Group Astronomers))
        
        %% Human clicks a button
        Astro --> |Clicks Interesting| PostFeedback[POST /api/annotations origin:CYOA-Human-*]
        PostFeedback --> FritzDB
    end

    %% Stage 5: Active Learning Retraining
    subgraph S5 [5. Active Learning Loop]
        FritzDB --> |GET /api/sources/annotations| CronJob[retrain.py cron job]
        CronJob --> |Pulls CYOA-Human Feedback| Updater[Update Scikit-Learn Model]
        Updater -.-> |Weights heavily in tomorrow's Triage| Profile
    end

    %% Styling
    classDef target fill:#4a90e2,stroke:#333,stroke-width:2px,color:#fff;
    classDef base layer fill:#2c3e50,stroke:#333,stroke-width:2px,color:#fff;
    classDef frontend fill:#e67e22,stroke:#333,stroke-width:2px,color:#fff;
    classDef retrain fill:#27ae60,stroke:#333,stroke-width:3px,color:#fff;

    class Base,XGBoost,Delta base;
    class Profile,IF,LLM,Verdict,BoomAPI,AnnotAPI target;
    class UI,PostFeedback frontend;
    class CronJob,Updater retrain;
```

## How the Freezing Works
- The **gray boxes** (Delta, Base Vectors, XGBoost) represent the **Locked Pipeline**. They act entirely independent of human opinions.
- The **blue boxes** (Isolation Forest, LLM) represent our **Inference Filter**.
- The **orange boxes** (React Widget, SkyPortal APIs) represent the **Human Bridge**.
- The **green boxes** (`retrain.py`) represents the **Active Learning Layer**. 

Notice how `retrain.py` circles back strictly to the `Profile Classifiers`. If the Supernova Hunters click "Noise", it only updates their specific model weights, without damaging the core vectors or interfering with the Rare Transients group.
