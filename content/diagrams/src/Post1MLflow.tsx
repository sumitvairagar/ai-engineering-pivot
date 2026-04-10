import { AbsoluteFill, useCurrentFrame } from "remotion";
import React from "react";

const steps = [
  { icon: "🧪", title: "6 Experiment Runs", sub: "Order Fill Predictor · different hyperparams", gradient: "linear-gradient(135deg, #E3F2FD, #BBDEFB)" },
  { icon: "📊", title: "Tracked in MLflow", sub: "Accuracy, F1, learning rate, tree depth", gradient: "linear-gradient(135deg, #BBDEFB, #90CAF9)" },
  { icon: "🔍", title: "Compared Side-by-Side", sub: "One dashboard · all runs visible", gradient: "linear-gradient(135deg, #90CAF9, #64B5F6)" },
  { icon: "🚀", title: "Best → Production", sub: "One alias change · instant rollback", gradient: "linear-gradient(135deg, #64B5F6, #42A5F5)" },
];

const FlowingDotsVertical: React.FC<{ index: number }> = ({ index }) => {
  const frame = useCurrentFrame();
  const lineLen = 36;
  return (
    <div style={{ display: "flex", justifyContent: "center", height: 52 }}>
      <svg width="24" height="52" viewBox="0 0 24 52">
        <line x1="12" y1="4" x2="12" y2="40" stroke="#CFD8DC" strokeWidth="2" strokeLinecap="round" />
        {[0, 1, 2].map((d) => {
          const t = ((frame * 1.5 + d * (lineLen / 3)) % lineLen) / lineLen;
          const cy = 4 + t * lineLen;
          const opacity = t < 0.1 ? t / 0.1 : t > 0.9 ? (1 - t) / 0.1 : 1;
          return <circle key={d} cx={12} cy={cy} r={3} fill="#1565C0" opacity={opacity} />;
        })}
        <path d="M6 36 L12 46 L18 36" stroke="#1565C0" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    </div>
  );
};

export const Post1MLflow: React.FC = () => (
  <AbsoluteFill style={{ background: "linear-gradient(180deg, #FAFCFF 0%, #E8F0FE 50%, #F0F4FF 100%)", fontFamily: "'SF Pro Display', 'Inter', system-ui, sans-serif", justifyContent: "center", alignItems: "center", padding: 60 }}>
    <div style={{ fontSize: 28, fontWeight: 900, color: "#0D47A1", marginBottom: 48, textAlign: "center", letterSpacing: -0.5, lineHeight: 1.35 }}>
      6 experiments on brokerage data<br />tracked, compared, deployed
    </div>

    <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
      {steps.map((step, i) => (
        <React.Fragment key={i}>
          <div style={{ width: 420 }}>
            <div style={{
              background: step.gradient, borderRadius: 18, padding: "20px 24px", textAlign: "center",
              boxShadow: "0 6px 24px rgba(0,0,0,0.08)", border: "1px solid rgba(255,255,255,0.7)",
            }}>
              <div style={{ fontSize: 32 }}>{step.icon}</div>
              <div style={{ fontSize: 18, fontWeight: 800, marginTop: 6, color: "#0D47A1" }}>{step.title}</div>
              <div style={{ fontSize: 13, color: "#37474F", marginTop: 4, fontStyle: "italic" }}>{step.sub}</div>
            </div>
          </div>
          {i < steps.length - 1 && <FlowingDotsVertical index={i} />}
        </React.Fragment>
      ))}
    </div>

    <div style={{ marginTop: 40, display: "flex", gap: 20, alignItems: "center" }}>
      <span style={{ fontSize: 14, color: "#78909C", fontWeight: 500 }}>MLflow + scikit-learn</span>
      <span style={{ fontSize: 14, color: "#B0BEC5" }}>·</span>
      <span style={{ fontSize: 14, color: "#1565C0", fontWeight: 700 }}>Sumit Vairagar</span>
    </div>
  </AbsoluteFill>
);
