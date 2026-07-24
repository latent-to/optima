const renderOptimaDiagrams = async () => {
  const blocks = Array.from(document.querySelectorAll("pre.mermaid"));
  if (!blocks.length) return;

  // SuperFences emits <pre><code>; Mermaid expects the diagram text directly
  // inside its target node. Normalize once so instant navigation cannot run a
  // second render over an already-generated SVG.
  const nodes = blocks.map((block) => {
    const node = document.createElement("div");
    node.className = "mermaid";
    node.textContent = block.textContent;
    block.replaceWith(node);
    return node;
  });

  try {
    const dark = document.body.getAttribute("data-md-color-scheme") === "slate";
    mermaid.initialize({
      startOnLoad: false,
      securityLevel: "loose",
      theme: dark ? "dark" : "base",
      themeVariables: {
        primaryColor: dark ? "#17213a" : "#e8fbf7",
        primaryTextColor: dark ? "#eef4ff" : "#10182a",
        primaryBorderColor: "#43d9bd",
        lineColor: dark ? "#8794b6" : "#53617d",
        secondaryColor: dark ? "#24214a" : "#eeefff",
        tertiaryColor: dark ? "#11182a" : "#f7f9fc",
        fontFamily: "Inter, ui-sans-serif, system-ui, sans-serif"
      },
      flowchart: { curve: "basis", htmlLabels: true }
    });
    await mermaid.run({ nodes });
  } catch (error) {
    const message = error && error.message ? error.message : String(error);
    console.error(`Optima diagram render failed: ${message}`);
  }
};

if (typeof document$ !== "undefined") {
  document$.subscribe(() => void renderOptimaDiagrams());
} else {
  document.addEventListener("DOMContentLoaded", () => void renderOptimaDiagrams());
}
