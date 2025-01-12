pipeline {
    agent any

    stages {
        stage('Checkout Code') {
            steps {
                checkout scm
            }
        }

        stage('Build Docker Image') {
            steps {
                script {
                    docker.build("mlops-app:latest", ".")
                }
            }
        }

        stage('Run Unit Tests') {
            steps {
                sh 'pytest tests/'
            }
        }

        stage('Push Docker Image') {
            steps {
                script {
                    docker.withRegistry('https://index.docker.io/v1/', 'docker-hub-credentials') {
                        docker.image("mlops-app:latest").push("latest")
                    }
                }
            }
        }

        stage('Deploy to Environment') {
            steps {
                sh 'docker-compose up -d'
            }
        }
    }

    post {
        always {
            archiveArtifacts artifacts: '**/results/*', allowEmptyArchive: true
        }
        success {
            echo 'Build, test, and deployment completed successfully!'
        }
        failure {
            echo 'Pipeline failed. Check logs.'
        }
    }
}
