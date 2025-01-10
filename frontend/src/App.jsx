import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [text, setText] = useState("");
  const [modelName, setModelName] = useState("spacy");
  const [entities, setEntities] = useState([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async () => {
    setError("");
    setEntities([]);
    setLoading(true);

    try {
      const response = await axios.post("http://213.171.27.97:8000/predict", {
        text: text,
        model_name: modelName
      });
      if (response.data && response.data.entities) {
        setEntities(response.data.entities);
      }
    } catch (err) {
      console.error(err);
      setError("Ошибка при запросе к серверу");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header>
        <h1>NER Stand</h1>
      </header>

      <main>
        <section className="input-section">
          <div className="form-group">
            <label>Текст для анализа:</label>
            <textarea
              rows={5}
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Введите любой текст на русском (или другом языке) для NER"
            />
          </div>

          <div className="form-group">
            <label>Выберите модель:</label>
            <select value={modelName} onChange={(e) => setModelName(e.target.value)}>
              <option value="spacy">spaCy (ru_core_news_sm)</option>
              <option value="hf">HuggingFace (BERT)</option>
              <option value="flair">Flair (NER-fast)</option>
            </select>
          </div>

          <button
            className="submit-button"
            onClick={handleSubmit}
            disabled={loading}
          >
            {loading ? "Загрузка..." : "Анализировать"}
          </button>
        </section>

        {error && <p className="error-message">{error}</p>}

        <section className="results-section">
          <h2>Найденные сущности</h2>
          {entities.length === 0 ? (
            <p className="no-entities">Пока нет сущностей</p>
          ) : (
            <div className="entities-list">
              {entities.map((ent, idx) => (
                <div key={idx} className="entity-card">
                  <span className="entity-type">{ent.entity}</span>
                  <div className="entity-text">«{ent.text}»</div>
                  <div className="entity-offset">
                    [start={ent.start_offset}, end={ent.end_offset}]
                  </div>
                </div>
              ))}
            </div>
          )}
        </section>
      </main>

      <footer>
        <p>© 2025. NER Stand.</p>
      </footer>
    </div>
  );
}

export default App;