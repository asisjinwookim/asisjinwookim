(load "../../modules/3/stream-advanced.ss")


(define (integral integrand initial-value dt)
  (define int
    (cons-stream initial-value
		 (add-streams (scale-stream integrand dt)
			      int)))
  int)
;; More study Needed
