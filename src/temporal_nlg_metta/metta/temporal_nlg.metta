;; ------------------------------------------------------------------
;; temporal_nlg.metta
;;
;; Kernel-portable MeTTa wrappers for the temporal explanation bridge.
;;
;; These definitions are written in pure MeTTa (no Python grounded atoms) so
;; they evaluate identically on the `hyperon` interpreter and on the MORK
;; kernel. The grounded operations (graph-answer, nlg-fact, tms-*, ...) are
;; provided by the Python bridge and must be registered before this file is
;; used on `hyperon`; MORK only consumes the pure-MeTTa helpers here.
;;
;; Convention: every grounded op returns its structured result as a JSON string.
;; The helpers below are thin compositional conveniences over those JSON blobs.
;; ------------------------------------------------------------------

;; A self-documenting registry of the grounded tokens this file expects to
;; interact with. Declared as plain symbols so the file loads even when the
;; Python bridge is absent (each is only invoked at call time).
(: temporal-token-registry Type)
(= temporal-token-registry
   (graph-answer graph-explain-path graph-mermaid graph-confidence graph-evidence
     nlg-fact nlg-fact-strategy nlg-readability nlg-eval
     tms-add-belief tms-record-rule tms-explain tms-contradictions
     tms-support-chain tms-retract! tms-start-trace tms-rules-fired))

;; Convenience: ask a temporal question and immediately render its evidence
;; graph (answer text + mermaid). Returns a small MeTTa struct.
(= (temporal-QA $question)
   (let* (($answer (graph-answer $question))
          ($mermaid (graph-mermaid)))
     (Answer $answer $mermaid)))

;; Compose a justification for a belief already recorded by the bridge.
(= (justify $belief-id)
   (Justification $belief-id (tms-explain $belief-id)))

;; Summarize the current contradiction status of the session.
(= (contradiction-status)
   (Contradictions (tms-contradictions)))
