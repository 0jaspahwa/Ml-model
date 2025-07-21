import React, { useState } from "react";
import axios from "axios";
import HeroSection from "./components/HeroSection";
import MetricsPanel from "./components/MetricsPanel";

function App() {
  const [personalization, setPersonalization] = useState(null);

  const handleFetch = async () => {
    try {
      const res = await axios.post("http://127.0.0.1:5000/personalize", {
        user_input: {
          user_id: "ojas_001",
          device_type: "mobile",
          region: "South"
        },
        real_time_behavior: []
      });

      if (res.data.status === "success") {
        setPersonalization(res.data.data); // contains hero, predictions, metadata
      } else {
        console.error("Personalization failed:", res.data.message);
      }
    } catch (err) {
      console.error("API error:", err);
    }
  };

  return (
    <div className="App">
      <h1>Ultra AI Personalization Engine</h1>

      <button onClick={handleFetch}>🎯 Generate Personalization</button>

      {personalization && (
        <>
          <HeroSection
            content={personalization.personalized_content.hero_section}
            strength={personalization.personalized_content.personalization_strength}
          />

          <MetricsPanel
            predictions={personalization.ai_predictions}
            version={personalization.metadata.ai_model_version}
          />
        </>
      )}
    </div>
  );
}

export default App;
