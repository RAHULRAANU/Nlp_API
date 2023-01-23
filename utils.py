from contextlib import contextmanager
from threading import Semaphore
from fastapi import HTTPException


class RequestLimiter:
    def __init__(self, limit):
        self.semaphore = Semaphore(limit - 1)

    @contextmanager
    def run(self):
        acquired = self.semaphore.acquire(blocking=False)
        if not acquired:
            raise HTTPException(
                status_code=503, detail="The server is busy processing requests."
            )
        try:
            yield acquired
        finally:
            self.semaphore.release()


# for utils
# from fastapi import Request
# concurrency_limiter = RequestLimiter(CONCURRENT_REQUEST_PER_WORKER)
#
# from data_validation import Textrequest
#
# @router.post("", response_model=Textrequest)
# def query(payload: Textrequest, request: Request):
#     model = request.app.state.spacy_ner_model
#
#     with concurrency_limiter.run():
#         return model.predict(payload)




