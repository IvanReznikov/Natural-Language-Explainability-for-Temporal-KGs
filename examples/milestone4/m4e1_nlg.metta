;; ------------------------------------------------------------------
;; Milestone 4 — Example 1: Natural-language generation from MeTTa (M1)
;;
;; Demonstrates the M1 NLG surface (template/polish/llm routing, readability,
;; accuracy evaluation) invoked from a MeTTa program. The grounded operations
;; nlg-fact, nlg-fact-strategy, nlg-readability and nlg-eval are registered by
;; the Python runner that loads this file (see m4e1_nlg.py).
;;
;; Each op returns a JSON string so structured results pass cleanly through
;; MeTTa atoms.
;; ------------------------------------------------------------------

;; Point-in-time fact verbalized with the default (auto-routed) strategy.
!(nlg-fact "point_in_time"
           "{\"entity\":\"Albert Einstein\",\"event\":\"was born\",\"date\":\"1879\"}")

;; Same fact forced through the fast template path.
!(nlg-fact-strategy "point_in_time"
                    "{\"entity\":\"Albert Einstein\",\"event\":\"was born\",\"date\":\"1879\"}"
                    "template")

;; Readability of a generated sentence.
!(nlg-readability "Albert Einstein was born in 1879.")

;; Accuracy of a candidate sentence against the source fact.
!(nlg-eval "{\"entity\":\"Albert Einstein\",\"event\":\"was born\",\"date\":\"1879\"}"
           "Albert Einstein was born in 1879."
           "point_in_time")
