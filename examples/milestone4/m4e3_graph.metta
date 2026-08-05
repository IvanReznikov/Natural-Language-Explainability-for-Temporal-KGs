;; ------------------------------------------------------------------
;; Milestone 4 — Example 3: Temporal graph QA from MeTTa (M3)
;;
;; Demonstrates the M3 surface: asking a temporal question over the knowledge
;; graph and reading back the natural-language answer, the evidence edges, the
;; confidence, and the Mermaid visualization of the supporting subgraph.
;; ------------------------------------------------------------------

;; Ask and answer a temporal causal question.
!(graph-answer "What likely caused the Model T price drop in 1913?")

;; Cheap accessors over the most recent answer (no re-retrieval).
!(graph-confidence)
!(graph-evidence)
!(graph-mermaid)

;; A second question — existence/intent this time.
!(graph-answer "Was Model T connected to Ford production scaling in 1913?")
!(graph-confidence)
