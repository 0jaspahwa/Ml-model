import React from "react";

function ControlsPanel({ userInput, setUserInput, onGenerate }) {
  const handleChange = (e) => {
    setUserInput({ ...userInput, [e.target.name]: e.target.value });
  };

  return (
    <div className="controls-panel">
      <h2>🎯 Personalization Controls</h2>
      <div className="control-group">
        <label>User ID</label>
        <input type="text" name="user_id" value={userInput.user_id} onChange={handleChange} />

        <label>Device Type</label>
        <select name="device_type" value={userInput.device_type} onChange={handleChange}>
          <option value="desktop">🖥️ Desktop</option>
          <option value="mobile">📱 Mobile</option>
          <option value="tablet">📱 Tablet</option>
        </select>

        <label>Region</label>
        <select name="region" value={userInput.region} onChange={handleChange}>
          <option value="US">🇺🇸 United States</option>
          <option value="EU">🇪🇺 Europe</option>
          <option value="ASIA">🌏 Asia</option>
          <option value="OTHER">🌍 Other</option>
        </select>

        <label>Age Group</label>
        <select name="age_group" value={userInput.age_group} onChange={handleChange}>
          <option value="18-25">18-25</option>
          <option value="26-35">26-35</option>
          <option value="36-45">36-45</option>
          <option value="46-55">46-55</option>
          <option value="55+">55+</option>
        </select>

        <label>Simulated Behavior</label>
        <select name="behaviorType" value={userInput.behaviorType} onChange={handleChange}>
          <option value="browsing">🔍 Casual Browsing</option>
          <option value="item_view">👁️ Product Viewing</option>
          <option value="add_to_cart">🛒 Add to Cart</option>
          <option value="search">🔎 Active Searching</option>
          <option value="purchase">💳 Purchase</option>
        </select>
      </div>

      <button className="generate-btn" onClick={onGenerate}>
        🚀 Generate AI Personalization
      </button>
    </div>
  );
}

export default ControlsPanel;
