;; ------------------------------------------------------------------
;; Milestone 4 — Example 8: Temporal reasoning IN MeTTa (not just from it)
;;
;; The other examples call the Python-backed capabilities from MeTTa. This one
;; moves a step of the reasoning itself into the MeTTa evaluator:
;;
;;   1. Temporal edges are injected as matchable *atoms* — not opaque JSON
;;      strings — via the `edges-from-json` grounded op.
;;   2. MeTTa's own pattern matcher (`match`) derives the causal link from the
;;      atomspace. The inference below is performed by the MeTTa evaluator.
;;   3. Only that MeTTa-derived conclusion is recorded into the M2
;;      truth-maintained trace (via `tms-record-rule-facts`, which takes plain
;;      atoms instead of a hand-built JSON string).
;;
;; So the recorded justification answers "why was this concluded?" about an
;; inference the MeTTa program actually made — the trace input is the output
;; of `match`, not a literal scripted in advance.
;;
;; Scenario: Model T causality edges (1913-1914).
;; ------------------------------------------------------------------

!(tms-start-trace "q-metta-reasoning")

;; Inject temporal edges as atoms: (edge <src> <rel> <tgt> <time>).
;; `edges-from-json` yields one result per edge; `let` + `add-atom` asserts
;; each into the runner's atomspace.
!(let $e (edges-from-json "[[\"moving assembly line\",\"caused\",\"Model T price drop\",\"1913\"],[\"Ford\",\"introduced\",\"moving assembly line\",\"1913\"],[\"Model T price drop\",\"reduced\",\"car prices\",\"1914\"]]")
     (add-atom &self $e))

;; MeTTa-side reasoning step: pattern-match the causal edge in the atomspace.
;; This is the derivation — performed by the MeTTa evaluator itself.
!(match &self (edge $s "caused" $t $y) (causal-link $s $t $y))

;; Record ONLY what MeTTa derived into the M2 trace: the rule firing's input
;; and conclusion come from the match bindings, not from literals.
!(let (causal-link $s $t $y)
       (match &self (edge $s2 "caused" $t2 $y2) (causal-link $s2 $t2 $y2))
       (tms-record-rule-facts "rule:causal-chain-in-metta"
                              "Derived by MeTTa pattern match over temporal edges"
                              $s $y $t "causal" 0.9))

;; The M2 debugging surface now explains the MeTTa-derived inference.
!(tms-rules-fired)
!(tms-explain-trace "Model T price drop")
