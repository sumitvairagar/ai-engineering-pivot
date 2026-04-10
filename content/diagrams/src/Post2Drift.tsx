import { AbsoluteFill, useCurrentFrame } from "remotion";
import React from "react";

const steps = [
  { icon: "📈", title: "Model Trained", sub: "On last quarter's brokerage data", gradient: "linear-gradient(135deg, #E8F5E9, #C8E6C9)", accent: "#2E7D32" },
  { icon: "✅", title: "Deployed to Production", sub: "Predicting order fills", gradient: "linear-gradient(135deg, #C8E6C9, #A5D6A7)", accent: "#2E7D32" },
  { icon: "🌪️", title: "Market Conditions Change", sub: "Volatility spikes · new patterns", gradient: "linear-gradient(135deg, #FFF9C4, #FFF176)", accent: "#F57F17" },
  { icon: "⚠️", title: "Drift Detected!", sub: "2 of 3 input features shifted", gradient: "linear-gradient(135deg, #FFCDD2, #EF9A9A)", accent: "#C62828" },
  { icon: "🔄", title: "Retrain — Data-Driven", sub: "Not guesswork · evidence-based", gradient: "linear-gradient(135deg, #E8F5E9, #A5D6A7)", accent: "#2E7D32" },
];

const FlowingDotsVertical: React.FC<{ index: number; danger?: boolean }> = ({ index, danger }) => {
  const frame = useCurrentFrame();
  const lineLen = 30;
  const color = danger ? "#C62828" : "#2E7D32";
  return (
    <div style={{ display: "flex", justifyContent: "center", height: 44 }}>
      <svg width="24" height="44" viewBox="0 0 24 44">
        <line x1="12" y1="4" x2="12" y2="34" stroke="#CFD8DC" strokeWidth="2" strokeLinecap="round" />
        {[0, 1, 2].map((d) => {
          const t = ((frame * 1.5 + d * (lineLen / 3)) % lineLen) / lineLen;
          const cy = 4 + t * lineLen;
          const opacity = t < 0.1 ? t / 0.1 : t > 0.9 ? (1 - t) / 0.1 : 1;
          return <circle key={d} cx={12} cy={cy} r={2.5} fill={color} opacity={opacity} />;
        })}
        <path d="M6 30 L12 39 L18 30" stroke={color} strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    </div>
  );
};

export const Post2Drift: React.FC = () => {
  const frame = useCurrentFrame();
  const glow = 8 + 5 * Math.sin(frame * 0.1);

  return (
    <AbsoluteFill style={{ background: "linear-gradient(180deg, #FFFDF7 0%, #FFF8E8 30%, #FFF0F0 70%, #F5FFF5 100%)", fontFamily: "'SF Pro Display', 'Inter', system-ui, sans-serif", justifyContent: "center", alignItems: "center", padding: 50 }}>
      <div style={{ fontSize: 26, fontWeight: 900, color: "#B71C1C", marginBottom: 40, textAlign: "center", letterSpacing: -0.5, lineHeight: 1.35 }}>
        Your deployed model is decaying<br />— here's how to catch it
      </div>

      <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
        {steps.map((step, i) => {
          const isDrift = i === 3;
          return (
            <React.Fragment key={i}>
              <div style={{ width: 400 }}>
                <div style={{
                  background: step.gradient, borderRadius: 16, padding: "16px 22px", textAlign: "center",
                  boxShadow: isDrift ? `0 0 ${glow}px rgba(198,40,40,0.35), 0 4px 16px rgba(0,0,0,0.08)` : "0 4px 16px rgba(0,0,0,0.07)",
                  border: isDrift ? "2px solid #C62828" : "1px solid rgba(255,255,255,0.6)",
                }}>
                  <div style={{ fontSize: 28 }}>{step.icon}</div>
                  <div style={{ fontSize: 16, fontWeight: 800, marginTop: 5, color: step.accent }}>{step.title}</div>
                  <div style={{ fontSize: 12, color: "#455A64", marginTop: 3, fontStyle: "italic" }}>{step.sub}</div>
                </div>
              </div>
              {i < steps.length - 1 && <FlowingDotsVertical index={i} danger={i >= 2} />}
            </React.Fragment>
          );
        })}
      </div>

      <div style={{ marginTop: 32, display: "flex", gap: 20, alignItems: "center" }}>
        <span style={{ fontSize: 14, color: "#78909C", fontWeight: 500 }}>Evidently AI + scikit-learn</span>
        <span style={{ fontSize: 14, color: "#B0BEC5" }}>·</span>
        <span style={{ fontSize: 14, color: "#C62828", fontWeight: 700 }}>Sumit Vairagar</span>
      </div>
    </AbsoluteFill>
  );
};
