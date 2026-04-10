import { Composition } from "remotion";
import { Post1MLflow } from "./Post1MLflow";
import { Post2Drift } from "./Post2Drift";

// 1080x1080 = LinkedIn square format, great for mobile
export const RemotionRoot = () => (
  <>
    <Composition id="Post1MLflow" component={Post1MLflow} durationInFrames={150} fps={30} width={1080} height={1080} />
    <Composition id="Post2Drift" component={Post2Drift} durationInFrames={180} fps={30} width={1080} height={1080} />
  </>
);
