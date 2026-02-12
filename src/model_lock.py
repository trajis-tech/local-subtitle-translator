"""
模型互斥鎖機制，確保任何時刻只載入一個模型。

使用 threading.Lock 保護模型載入/卸載與推理；若併發觸發兩個流程，
hold_model() 會讓後者阻塞直到前者完整卸載並釋放鎖，避免同時存在兩模型。
"""

from __future__ import annotations
import threading
import gc
from typing import Optional, Any


class _HoldModelContext:
    """Context manager 內部實作：進入時 acquire + 設定 model_type；離開時由呼叫端確保已 del + gc，再 set None 並 release。"""

    def __init__(self, mutex: ModelMutex, model_type: str):
        self._mutex = mutex
        self._model_type = model_type

    def __enter__(self):
        self._mutex._mutex.acquire()
        self._mutex.set_current_model_type(self._model_type)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._mutex.set_current_model_type(None)
        self._mutex._mutex.release()
        return False  # 不吞掉例外


class ModelMutex:
    """
    模型互斥鎖，確保任何時刻只有一個模型實例存在。

    建議使用 hold_model(model_type)，將「載入 → 推理 → 卸載」整段包在鎖內，
    避免僅鎖 load/unload 而推理在鎖外導致併發時同時存在兩模型。

    使用方式：
        with model_mutex.hold_model("reason"):
            model = load_model(...)
            # 推理
            del model
            gc.collect()
        # 離開時自動 set_current_model_type(None) 並 release
    """

    _instance: Optional[ModelMutex] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._mutex = threading.Lock()
                    cls._instance._current_model_type: Optional[str] = None
        return cls._instance

    def __enter__(self):
        """進入互斥區（僅鎖，不設定 model_type；新程式碼請用 hold_model）。"""
        self._mutex.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """離開互斥區"""
        self._mutex.release()
        return False

    def hold_model(self, model_type: str) -> _HoldModelContext:
        """
        取得「持鎖 + 設定當前模型類型」的 context manager。
        進入時：acquire lock + set_current_model_type(model_type)。
        離開時：由呼叫端在 block 內確保已 del model、gc.collect()，再離開；
        離開時會 set_current_model_type(None) 並 release lock。
        若未來有併發，可明確阻止同時載入兩模型（後者會阻塞直到前者完整卸載）。
        """
        return _HoldModelContext(self, model_type)

    def set_current_model_type(self, model_type: Optional[str]):
        """設置當前模型類型（用於調試）"""
        self._current_model_type = model_type

    def get_current_model_type(self) -> Optional[str]:
        """取得當前模型類型（用於調試）"""
        return self._current_model_type


# 全域單例
_model_mutex = ModelMutex()


def get_model_mutex() -> ModelMutex:
    """取得全域模型互斥鎖"""
    return _model_mutex


def ensure_model_unloaded(log_lines: Optional[list[str]] = None):
    """
    確保沒有模型被載入（強制清理）。

    Args:
        log_lines: 可選的 log 列表（用於記錄）
    """
    with _model_mutex:
        gc.collect()
        if log_lines is not None:
            log_lines.append("[ModelMutex] Ensured all models are unloaded")


# ----- 自測：單執行緒 hold_model 設/清 type；併發時第二者會阻塞 -----
if __name__ == "__main__":
    mutex = get_model_mutex()
    assert mutex.get_current_model_type() is None
    with mutex.hold_model("reason"):
        assert mutex.get_current_model_type() == "reason"
    assert mutex.get_current_model_type() is None
    print("[OK] hold_model single-thread: type set on enter, cleared on exit")

    result = {"second_entered_after_first_exited": False}

    def first_holder():
        with mutex.hold_model("reason"):
            import time
            time.sleep(0.15)
        result["first_exited"] = True

    def second_holder():
        import time
        time.sleep(0.05)  # 讓 first 先進入
        t0 = time.perf_counter()
        with mutex.hold_model("translate"):
            result["second_entered_after_first_exited"] = result.get("first_exited", False)
        result["second_held_duration"] = time.perf_counter() - t0

    t1 = threading.Thread(target=first_holder)
    t2 = threading.Thread(target=second_holder)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    # 第二個 thread 應在第一個 exit 後才進入，故 second_entered_after_first_exited 為 True
    assert result.get("second_entered_after_first_exited"), "Concurrent hold_model: second should block until first exits"
    print("[OK] hold_model concurrent: second thread blocked until first released")
