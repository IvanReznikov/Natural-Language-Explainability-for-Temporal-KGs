;; ------------------------------------------------------------------
;; Milestone 4 — Example 5: Multi-hop justification paths (M2 depth)
;;
;; Demonstrates full-chain justification extraction: given a multi-step
;; derivation recorded into the trace, reconstruct the *entire chain of rules*
;; that produced a conclusion. This is the signature capability for
;; "accelerate theorem-prover debugging" — instead of a single rationale, you
;; get the full reasoning path from premises to conclusion.
;;
;; Scenario: a 3-step derivation
;;   raw_record -> year_extracted -> year_normalized -> birth_year
;; ------------------------------------------------------------------

!(tms-start-trace "q-justify")

;; Step 1: raw census record -> extracted year
!(tms-record-rule "rule:extract-year"
                  "Extract year from census record"
                  "[{\"fact_id\":\"raw_record\",\"value\":\"born 1879\"}]"
                  "{\"fact_id\":\"year_extracted\",\"value\":\"1879\"}"
                  0.95)

;; Step 2: extracted year -> normalized ISO year (depends on year_extracted)
!(tms-record-rule "rule:normalize-year"
                  "Normalize year to ISO format"
                  "[{\"fact_id\":\"year_extracted\",\"value\":\"1879\"}]"
                  "{\"fact_id\":\"year_normalized\",\"value\":\"1879-01-01\"}"
                  0.92)

;; Step 3: normalized year -> conclusion birth_year (depends on year_normalized)
!(tms-record-rule "rule:assert-birth-year"
                  "Assert birth year conclusion"
                  "[{\"fact_id\":\"year_normalized\",\"value\":\"1879-01-01\"}]"
                  "{\"fact_id\":\"birth_year\",\"value\":\"1879\"}"
                  0.9)

;; Now extract the full justification chain for the conclusion.
;; This walks the dependency graph backwards: birth_year <- year_normalized <- year_extracted <- raw_record
!(tms-justification-path "birth_year")

;; Also show a single-hop justification for contrast.
!(tms-explain-trace "birth_year")
