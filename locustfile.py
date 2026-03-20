from locust import HttpUser, task, between


class MLApiUser(HttpUser):
    wait_time = between(1, 2)  # Время между запросами
    model_id_lr = None
    model_id_rf = None

    def on_start(self):
        # Обучение модели Logistic Regression
        response_lr = self.client.post(
            "/api/v1/models/train",
            json={
                "model_type": "logistic_regression",
                "dataset_name": None,
                "hyperparameters": {}
            }
        )
        if response_lr.status_code == 200:
            body_lr = response_lr.json()
            self.model_id_lr = body_lr.get("model_id")

        # Обучение модели Random Forest
        response_rf = self.client.post(
            "/api/v1/models/train",
            json={
                "model_type": "random_forest",  # Обучение модели Random Forest
                "dataset_name": None,
                "hyperparameters": {}
            }
        )
        if response_rf.status_code == 200:
            body_rf = response_rf.json()
            self.model_id_rf = body_rf.get("model_id")

    @task(3)
    def health(self):
        """Проверка здоровья сервиса"""
        self.client.get("/api/v1/health")

    @task(2)
    def list_models(self):
        """Запрос на получение списка доступных моделей"""
        self.client.get("/api/v1/models")

    @task(2)
    def list_datasets(self):
        """Запрос на получение списка датасетов"""
        self.client.get("/api/v1/datasets")

    @task(4)
    def predict_lr(self):
        """Инференс для Logistic Regression"""
        if not self.model_id_lr:
            return
        self.client.post(
            f"/api/v1/models/{self.model_id_lr}/predict",
            json={
                "data": [[5.1, 3.5, 1.4, 0.2]]
            }
        )

    @task(4)
    def predict_rf(self):
        """Инференс для Random Forest"""
        if not self.model_id_rf:
            return
        self.client.post(
            f"/api/v1/models/{self.model_id_rf}/predict",
            json={
                "data": [[5.1, 3.5, 1.4, 0.2]]  # Пример данных
            }
        )