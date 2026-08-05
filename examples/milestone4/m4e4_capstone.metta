;; ------------------------------------------------------------------
;; Milestone 4 — Example 4 (capstone): Explaining a temporal inference path
;;
;; This is the headline deliverable: a MeTTa program that performs a temporal
;; inference, records each step of the inference into a truth-maintained trace
;; (M2), grounds its conclusion in evidence retrieved from the graph (M3), and
;; verbalizes the whole inference path as a natural-language explanation (M1).
;;
;; Scenario: "What enabled the Model T price drop, and how confident are we?"
;;   1. (M3) Retrieve the causal chain from the graph.
;;   2. (M2 + MeTTa) Open a trace; assert the answer's evidence edges as atoms
;;      and let MeTTa's own pattern matcher derive the causal rule firing that
;;      is recorded — the trace input is derived, not scripted.
;;   3. (M2) Register the conclusion as a belief supported by the evidence.
;;   4. (M1) Verbalize the path; (M2) render a justification.
;; ------------------------------------------------------------------

;; ── Step 1 (M3): retrieve the causal chain over the graph ─────────────
!(graph-answer "What likely caused the Model T price drop in 1913?")
!(graph-confidence)

;; ── Step 2 (M2 + MeTTa): derive the inference record from the evidence ─
!(tms-start-trace)
;; Assert the answer's evidence edges as matchable atoms...
!(let $e (graph-evidence-atoms) (add-atom &self $e))
;; ...and let MeTTa derive the causal edge(s); only derived firings are traced.
!(let (causal-link $s $t $y)
       (match &self (edge $s2 "caused" $t2 $y2) (causal-link $s2 $t2 $y2))
       (tms-record-rule-facts "rule:causal-chain-inference"
                              "Causal edge derived by MeTTa match over graph evidence"
                              $s $y $t "causal" 0.85))

;; ── Step 3 (M2): register the conclusion as an evidenced belief ───────
!(tms-add-belief "conclusion:model-t-price-drop"
                 "{\"claim\":\"The moving assembly line caused the Model T price drop in 1913\"}"
                 "[]"
                 "[{\"source\":\"graph\",\"snippet\":\"moving assembly line -> price drop\",\"weight\":0.8}]")

;; ── Step 4 (M1 + M2): verbalize + justify ─────────────────────────────
;; Verbalize the conclusion fact in natural language...
!(nlg-fact "causality"
           "{\"cause\":\"the moving assembly line\",\"effect\":\"the Model T price drop\",\"temporal_relation\":\"in 1913\"}"
           "template")
;; ...and render its truth-maintained justification.
!(tms-explain "conclusion:model-t-price-drop")
!(tms-support-chain "conclusion:model-t-price-drop")
!(tms-contradictions)
