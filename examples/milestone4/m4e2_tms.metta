;; ------------------------------------------------------------------
;; Milestone 4 — Example 2: Truth maintenance from MeTTa (M2)
;;
;; Demonstrates the M2 TMS surface: trace recording, belief registration with
;; supports/evidence, justification rendering, contradiction detection, the
;; "why-not-fired" debugging meta-query, and belief retraction with dirty-mark
;; propagation. This is the substrate that improves interpreter correctness and
;; accelerates theorem-prover debugging.
;; ------------------------------------------------------------------

;; Begin a traced reasoning session.
!(tms-start-trace)

;; Two rules that fire and both conclude about fact "f1" — the second with a
;; conflicting value, which the contradiction detector will surface below.
!(tms-record-rule "rule:derive-birth-year"
                  "Derive birth year from census"
                  "[{\"fact_id\":\"census_1879\",\"value\":\"1879\"}]"
                  "{\"fact_id\":\"f1\",\"value\":\"1879\"}"
                  0.9)

!(tms-record-rule "rule:alt-derivation"
                  "Alternate derivation with stale data"
                  "[{\"fact_id\":\"record_x\",\"value\":\"1880\"}]"
                  "{\"fact_id\":\"f1\",\"value\":\"1880\"}"
                  0.6)

;; Register beliefs with a support chain: B2 depends on B1.
!(tms-add-belief "B1"
                 "{\"claim\":\"Einstein was born in 1879\"}"
                 "[]"
                 "[{\"source\":\"census\",\"snippet\":\"f1=1879\",\"weight\":1.0}]")
!(tms-add-belief "B2"
                 "{\"claim\":\"Einstein published in Annalen der Physik\"}"
                 "[\"B1\"]")

;; Queries over the recorded trace + beliefs.
!(tms-rules-fired)
!(tms-contradictions)
!(tms-influential-facts)
!(tms-why-not "[\"rule:did-not-fire\"]")
!(tms-explain "B2")
!(tms-support-chain "B2")

;; Retract B1: dependents (B2) must be marked dirty — core TMS behavior.
!(tms-retract! "B1")
!(tms-dirty-beliefs)
