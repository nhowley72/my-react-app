import React, { useState } from "react";
import "./App.css";

function App() {
  const [number, setNumber] = useState("");
  const [result, setResult] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await fetch("http://localhost:8000/process-number/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ number: parseInt(number) }),
      });
      const data = await response.json();
      setResult(data.output);
    } catch (error) {
      console.error("Error:", error);
      setResult("Error connecting to server");
    }
  };

  return (
    <div className="app-container">
      <h1>Number Doubler</h1>
      <form onSubmit={handleSubmit} className="form-container">
        <input
          type="number"
          value={number}
          onChange={(e) => setNumber(e.target.value)}
          placeholder="Enter a number"
          className="input-field"
          required
        />
        <button type="submit" className="submit-button">Calculate</button>
      </form>
      {result !== null && <h2 className="result-display">Result: {result}</h2>}
    </div>
  );
}

export default App;
