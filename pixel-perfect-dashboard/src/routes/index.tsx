import { createFileRoute, Link } from "@tanstack/react-router";
import { ArrowRight, Leaf, Satellite, Sparkles } from "lucide-react";
import skyline from "@/assets/bengaluru-skyline.jpg";
import park from "@/assets/bengaluru-park.jpg";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "HeatLens — Bengaluru Urban Heat Intelligence" },
      { name: "description", content: "Explore Bengaluru's urban heat risk, microclimates, and greener futures." },
      { property: "og:title", content: "HeatLens — Bengaluru Urban Heat Intelligence" },
      { property: "og:description", content: "Explore Bengaluru's urban heat risk, microclimates, and greener futures." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: Landing,
});

const pillars = [
  {
    icon: Satellite,
    title: "Landsat-derived observations",
    text: "Ward-level land-surface temperature and heat-pocket coverage from multi-year Landsat 8/9 observations.",
  },
  {
    icon: Sparkles,
    title: "Model-estimated future risk",
    text: "A trained classifier estimates the probability of high heat at each ward's next observed Landsat pass.",
  },
  {
    icon: Leaf,
    title: "Decision-support, not certainty",
    text: "Trend analysis, explainable AI, and a policy engine turn observations into grounded, cautious recommendations.",
  },
];

function Landing() {
  return (
    <main className="landing">
      <div className="landing-atmosphere">
        <img src={skyline} alt="Bengaluru skyline at dusk" width={1920} height={1024} />
      </div>

      <header className="landing-top">
        <div className="brand-mark"><span /><i /></div>
        <div className="landing-brand"><b>HEATLENS</b><small>COOLER CITIES. BRIGHTER FUTURES.</small></div>
        <Link to="/overview" className="landing-nav-cta">Open Platform <ArrowRight /></Link>
      </header>

      <section className="landing-hero">
        <p className="landing-kicker">Bengaluru urban heat intelligence</p>
        <h1>Read the city<br />through its <em>heat signal.</em></h1>
        <p className="landing-sub">
          A satellite-based view of Bengaluru's 369 wards — current conditions, historical
          persistence, model-estimated future risk, and decision-support recommendations,
          built from Landsat-derived observations.
        </p>
        <div className="landing-actions">
          <Link to="/overview" className="landing-cta-primary">Read the city's heat signal <ArrowRight /></Link>
          <Link to="/explore" className="landing-cta-secondary">Explore wards</Link>
        </div>
      </section>

      <section className="landing-pillars">
        {pillars.map(({ icon: Icon, title, text }) => (
          <article key={title}>
            <div className="landing-pillar-icon"><Icon /></div>
            <strong>{title}</strong>
            <p>{text}</p>
          </article>
        ))}
      </section>

      <footer className="landing-footer">
        <img src={park} alt="Tree-lined park in Bengaluru" width={1024} height={768} loading="lazy" />
        <div>
          <blockquote>“A cooler Bengaluru is a healthier, more equitable Bengaluru.”</blockquote>
          <span>Built on Landsat-derived observations · Not a real-time monitoring or weather-forecasting system</span>
        </div>
      </footer>
    </main>
  );
}
