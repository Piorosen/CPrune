import logging

class Logger:
    _instance = None  # Singleton 패턴 적용을 위한 클래스 변수

    @staticmethod
    def get_logger():
        """전역에서 사용할 수 있는 로거 인스턴스를 반환합니다."""
        if Logger._instance is None:
            Logger()
        return Logger._instance

    def __init__(self):
        """로거를 설정합니다. 이 메서드는 한 번만 실행됩니다."""
        if Logger._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            Logger._instance = self._setup_logger()

    def _setup_logger(self):
        """로거의 설정을 정의합니다."""
        # 로거 설정
        logger = logging.getLogger("GlobalLogger")
        logger.setLevel(logging.DEBUG)

        # 콘솔 핸들러 설정
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)

        # 로그 포맷 설정
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)

        # 파일 핸들러 설정 (원한다면)
        file_handler = logging.FileHandler('app.log')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

        # 핸들러를 로거에 추가
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

        return logger