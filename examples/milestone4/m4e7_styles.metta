;; ------------------------------------------------------------------
;; Milestone 4 — Example 7: Style/domain narrative adaptation (M1 + M3)
;;
;; Demonstrates that the SAME temporal inference path can be explained to
;; different audiences: a novice (plain language), a neutral reader, and an
;; expert — and across domains (general / medical / finance). This is a
;; tangible "interactive, human-understandable AI" deliverable.
;;
;; The adjacency encodes one path: Ford Model T --produced_with--> Assembly
;; line --enabled--> Price drop (1913). It is rendered three ways below.
;; ------------------------------------------------------------------

;; The shared path, as a JSON adjacency map.
;; (Bound as a literal argument to each call below.)

;; ── Style variation: same path, three registers ───────────────────────
!(graph-explain-path-styled "{\"ModelT\":[[\"AssemblyLine\",\"produced_with\",\"1913\"]],\"AssemblyLine\":[[\"PriceDrop\",\"enabled\",\"1913\"]]}"
                            "ModelT" "PriceDrop" "novice" "general")

!(graph-explain-path-styled "{\"ModelT\":[[\"AssemblyLine\",\"produced_with\",\"1913\"]],\"AssemblyLine\":[[\"PriceDrop\",\"enabled\",\"1913\"]]}"
                            "ModelT" "PriceDrop" "neutral" "general")

!(graph-explain-path-styled "{\"ModelT\":[[\"AssemblyLine\",\"produced_with\",\"1913\"]],\"AssemblyLine\":[[\"PriceDrop\",\"enabled\",\"1913\"]]}"
                            "ModelT" "PriceDrop" "expert" "general")

;; ── Domain variation: same path, three domains ────────────────────────
!(graph-explain-path-styled "{\"ModelT\":[[\"AssemblyLine\",\"produced_with\",\"1913\"]],\"AssemblyLine\":[[\"PriceDrop\",\"enabled\",\"1913\"]]}"
                            "ModelT" "PriceDrop" "neutral" "finance")
