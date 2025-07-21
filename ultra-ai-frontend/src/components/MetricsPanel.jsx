import React from "react";

function MetricsPanel({ predictions, version }) {
  if (!predictions) return null;

  return (
    <div style={styles.card}>
      <h3>📈 AI Metrics</h3>
      <ul style={styles.list}>
        <li><strong>🧠 User Cluster:</strong> {predictions.cluster_label}</li>
        <li><strong>🎯 Confidence Score:</strong> {Math.round(predictions.cluster_confidence * 100)}%</li>
        <li><strong>⚡ Inferred Intent:</strong> {predictions.inferred_intent || "N/A"}</li>
        <li><strong>🤖 Model Version:</strong> {version}</li>
      </ul>
    </div>
  );
}

const styles = {
  card: {
    background: "#fdfdfd",
    padding: "20px",
    borderRadius: "12px",
    marginTop: "20px",
    boxShadow: "0 2px 10px rgba(0,0,0,0.06)"
  },
  list: {
    listStyle: "none",
    padding: 0,
    marginTop: "1rem",
    lineHeight: "1.6"
  }
};

export default MetricsPanel;
