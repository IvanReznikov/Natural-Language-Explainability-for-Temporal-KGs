;; ------------------------------------------------------------------
;; Milestone 4 — Example 6: Counterfactual reasoning (M1 + M2 + M3)
;;
;; Demonstrates two counterfactual modes, both new to M4:
;;
;;  (a) Fact-level counterfactual: pair a factual temporal fact against an
;;      alternative and verbalize the divergence. The hallmark of causal
;;      explainability: "if instead X, the outcome would diverge."
;;
;;  (b) Belief-level temporal shift: given a belief recorded in the TMS, ask
;;      "if its timeframe shifted by delta, what would it become?" The
;;      counterfactual is registered as a new belief supporting the original,
;;      preserving the dependency chain.
;; ------------------------------------------------------------------

;; ── (a) Fact-level counterfactual ─────────────────────────────────────
;; Factual: the moving assembly line caused the Model T price drop in 1913.
;; Alternative: what if hand-production caused a price rise in 1913 instead?
!(counterfactual "moving assembly line"
                 "caused"
                 "Model T price drop"
                 "hand production"
                 "caused"
                 "price rise")

;; A temporal counterfactual: same actors, different timeframe.
!(counterfactual "Model T"
                 "launched in"
                 "1908"
                 "Model T"
                 "launched in"
                 "1920"
                 "1908"
                 "1920")

;; ── (b) Belief-level temporal shift (TMS-integrated) ──────────────────
;; First record the factual belief, then shift its time.
!(tms-add-belief "belief:assembly-line-1913"
                 "{\"claim\":\"Assembly line adopted in 1913\"}"
                 "[]"
                 "[{\"source\":\"graph\",\"snippet\":\"Ford 1913\",\"weight\":0.9}]")

;; Pose the temporal counterfactual: what if it had been 5 years earlier?
!(tms-counterfactual-shift "belief:assembly-line-1913" "5 years earlier")

;; The counterfactual belief now supports the original — verify the chain grew.
!(tms-support-chain "belief:assembly-line-1913")
!(tms-active-beliefs)
