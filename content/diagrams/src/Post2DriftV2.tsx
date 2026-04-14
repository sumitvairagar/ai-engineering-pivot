import { AbsoluteFill, useCurrentFrame } from "remotion";
import React from "react";

const steps = [
  { icon: "📈", title: "Model Trained", sub: "Last quarter's brokerage data", bg: "linear-gradient(135deg, #00C853, #69F0AE)", border: "#00C853", titleColor: "#fff" },
  { icon: "✅", title: "Deployed", sub: "Predicting order fills", bg: "linear-gradient(135deg, #2979FF, #82B1FF)", border: "#2979FF", titleColor: "#fff" },
  { icon: "🌪️", title: "Market Shifts", sub: "Volatility spikes · new patterns", bg: "linear-gradient(135deg, #FFD600, #FFF176)", border: "#FFD600", titleColor: "#333" },
  { icon: "🚨", title: "Drift Detected!", sub: "2 of 3 input features shifted", bg: "linear-gradient(135deg, #FF1744, #FF8A80)", border: "#FF1744", titleColor: "#fff" },
  { icon: "🔄", title: "Retrain", sub: "Data-driven · not guesswork", bg: "linear-gradient(135deg, #00E676, #B9F6CA)", border: "#00E676", titleColor: "#1B5E20" },
];

const GAP = 48;

const FlowingDots: React.FC<{ index: number; danger?: boolean }> = ({ index, danger }) => {
  const frame = useCurrentFrame();
  const lineLen = GAP - 10;
  const color = danger ? "#FF1744" : "#00E676";
  return (
    <div style={{ display: "flex", justifyContent: "center", height: GAP }}>
      <svg width="24" height={GAP} viewBox={`0 0 24 ${GAP}`}>
        <line x1="12" y1="2" x2="12" y2={GAP - 8} stroke="rgba(255,255,255,0.15)" strokeWidth="2" strokeLinecap="round" />
        {[0, 1, 2].map((d) => {
          const t = ((frame * 1.8 + d * (lineLen / 3)) % lineLen) / lineLen;
          const cy = 2 + t * lineLen;
          const opacity = t < 0.1 ? t / 0.1 : t > 0.9 ? (1 - t) / 0.1 : 1;
          return <circle key={d} cx={12} cy={cy} r={3.5} fill={color} opacity={opacity} />;
        })}
        <path d={`M6 ${GAP - 12} L12 ${GAP - 4} L18 ${GAP - 12}`} stroke={color} strokeWidth="2.5" fill="none" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    </div>
  );
};

export const Post2DriftV2: React.FC = () => {
  const frame = useCurrentFrame();
  const driftGlow = 12 + 8 * Math.sin(frame * 0.12);
  const driftScale = 1 + 0.02 * Math.sin(frame * 0.12);

  return (
    <AbsoluteFill style={{
      background: "linear-gradient(160deg, #0D1117 0%, #161B22 40%, #1A0A0A 70%, #0D1117 100%)",
      fontFamily: "'SF Pro Display', 'Inter', system-ui, sans-serif",
      justifyContent: "center", alignItems: "center", padding: 50,
    }}>
      {/* Decorative glow blobs */}
      <div style={{ position: "absolute", top: 60, left: 80, width: 200, height: 200, borderRadius: "50%", background: "radial-gradient(circle, rgba(0,230,118,0.08), transparent)", filter: "blur(40px)" }} />
      <div style={{ position: "absolute", bottom: 80, right: 60, width: 250, height: 250, borderRadius: "50%", background: "radial-gradient(circle, rgba(255,23,68,0.1), transparent)", filter: "blur(50px)" }} />

      <div style={{ fontSize: 28, fontWeight: 900, color: "#FF8A80", marginBottom: 44, textAlign: "center", letterSpacing: -0.5, lineHeight: 1.35 }}>
        Your deployed model is decaying<br />
        <span style={{ color: "#FF1744" }}>— here's how to catch it</span>
      </div>

      <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
        {steps.map((step, i) => {
          const isDrift = i === 3;
          return (
            <React.Fragment key={i}>
              <div style={{
                width: 380,
                transform: isDrift ? `scale(${driftScale})` : undefined,
              }}>
                <div style={{
                  background: step.bg, borderRadius: 18, padding: "18px 22px",
                  display: "flex", alignItems: "center", gap: 16,
                  boxShadow: isDrift
                    ? `0 0 ${driftGlow}px rgba(255,23,68,0.5), 0 4px 20px rgba(0,0,0,0.4)`
                    : `0 4px 20px rgba(0,0,0,0.3)`,
                  border: `2px solid ${step.border}`,
                }}>
                  <div style={{ fontSize: 36, flexShrink: 0 }}>{step.icon}</div>
                  <div>
                    <div style={{ fontSize: 17, fontWeight: 800, color: step.titleColor, letterSpacing: -0.3 }}>{step.title}</div>
                    <div style={{ fontSize: 12, color: "rgba(255,255,255,0.7)", marginTop: 3 }}>{step.sub}</div>
                  </div>
                </div>
              </div>
              {i < steps.length - 1 && <FlowingDots index={i} danger={i >= 2} />}
            </React.Fragment>
          );
        })}
      </div>

      <div style={{ marginTop: 36, display: "flex", gap: 20, alignItems: "center" }}>
        <span style={{ fontSize: 13, color: "rgba(255,255,255,0.4)", fontWeight: 500 }}>Evidently AI + scikit-learn</span>
        <span style={{ fontSize: 13, color: "rgba(255,255,255,0.2)" }}>·</span>
        <span style={{ fontSize: 13, color: "#FF8A80", fontWeight: 700 }}>Sumit Vairagar</span>
      </div>
    </AbsoluteFill>
  );
};
