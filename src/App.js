import React, { useState } from "react";

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
    }
  };

  return (
    <div>
      <h1>Number Doubler</h1>
      <form onSubmit={handleSubmit}>
        <input
          type="number"
          value={number}
          onChange={(e) => setNumber(e.target.value)}
          placeholder="Enter a number"
          required
        />
        <button type="submit">Send</button>
      </form>
      {result !== null && <h2>Result: {result}</h2>}
    </div>
  );
}

export default App;
