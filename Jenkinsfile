pipeline {
  agent any
  stages {
    stage('Checkout Code') {
      steps {
        git(url: 'https://github.com/mahmuudtolba/autoencoder.git', branch: 'main')
      }
    }

    stage('log') {
      steps {
        sh 'ls -la'
      }
    }

    stage('Build') {
      steps {
        sh 'docker build -f .'
      }
    }

    stage('Log into Dockerhub') {
      steps {
        sh 'docker login -u $DOCKER'
      }
    }

  }
}