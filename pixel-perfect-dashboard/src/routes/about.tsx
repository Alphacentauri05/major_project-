import { createFileRoute } from "@tanstack/react-router";
import { Menu, Satellite } from "lucide-react";
import { AppShell, useSidebarToggle } from "@/components/AppShell";

export const Route = createFileRoute("/about")({
  head: () => ({
    meta: [
      { title: "About — HeatLens" },
      { name: "description", content: "What HeatLens is, and what it is not." },
    ],
  }),
  component: AboutPage,
});

function AboutPage() {
  const toggleSidebar = useSidebarToggle();

  return (
    <AppShell>
      <header className="page-header">
        <button type="button" className="action-button mobile-menu" aria-label="Toggle navigation" onClick={toggleSidebar}><Menu /></button>
        <div>
          <h1>About HeatLens</h1>
          <p>A Bengaluru urban heat intelligence platform built on satellite observations.</p>
        </div>
      </header>

      <div className="about-grid">
        <section className="ward-panel">
          <div className="panel-heading"><div><h2><Satellite /> What this is</h2></div></div>
          <p className="about-text">
            HeatLens analyzes Landsat 8/9-derived land-surface temperature and heat-pocket
            coverage across Bengaluru's 369 wards. It combines ward-level heat-pocket detection,
            multi-temporal persistence analysis, a trained machine-learning classifier, explainable
            AI (SHAP), FDR-corrected historical trend analysis, and a rule-based policy
            decision-support engine.
          </p>
        </section>

        <section className="ward-panel">
          <div className="panel-heading"><div><h2>What this is not</h2></div></div>
          <ul className="about-list">
            <li>Not a real-time monitoring system — figures reflect the most recent processed Landsat observation, not live conditions.</li>
            <li>Not a weather forecasting service — the "future risk" figure is a model-estimated probability for the next observed/usable Landsat scene, not a calendar-date forecast.</li>
            <li>Not a population health or vulnerability model — this platform does not model demographics, health outcomes, or human exposure.</li>
            <li>Not a source of guaranteed intervention outcomes — Policy Engine recommendations are decision-support, and effect sizes are not estimated unless explicitly stated.</li>
          </ul>
        </section>

        <section className="ward-panel">
          <div className="panel-heading"><div><h2>Data & methodology</h2></div></div>
          <p className="about-text">
            Ward boundaries come from the official 369-ward Greater Bengaluru Area boundary
            dataset. Every figure shown across Overview, Explore, Ward Intelligence, HeatLens AI,
            and PDF reports is read from the same underlying analytics pipeline — there is a single
            canonical source of truth per ward, so these surfaces should never disagree.
          </p>
        </section>
      </div>
    </AppShell>
  );
}
