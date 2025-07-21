import React from "react";

function HeroSection({ content, strength }) {
  return (
    <div className="hero-section" style={styles.container}>
      <h2 style={styles.title}>{content.title}</h2>
      <p style={styles.subtitle}>{content.subtitle}</p>
      <div style={styles.meta}>
        <span style={styles.badge}>✨ Personalization: {strength}</span>
        {content.cta && (
          <button style={styles.cta}>{content.cta}</button>
        )}
      </div>
    </div>
  );
}

const styles = {
  container: {
    background: "#f0f4ff",
    padding: "24px",
    borderRadius: "12px",
    marginTop: "20px",
    marginBottom: "20px",
    boxShadow: "0 2px 10px rgba(0,0,0,0.05)",
    textAlign: "center"
  },
  title: {
    fontSize: "1.8rem",
    fontWeight: "600",
    marginBottom: "0.5rem"
  },
  subtitle: {
    fontSize: "1rem",
    color: "#444",
    marginBottom: "1rem"
  },
  badge: {
    fontSize: "0.85rem",
    backgroundColor: "#d1d5db",
    padding: "6px 12px",
    borderRadius: "999px",
    marginRight: "10px",
    fontWeight: "bold"
  },
  cta: {
    background: "#4f46e5",
    color: "white",
    border: "none",
    padding: "10px 20px",
    borderRadius: "25px",
    cursor: "pointer",
    fontSize: "1rem",
    fontWeight: "600"
  }
};

export default HeroSection;
