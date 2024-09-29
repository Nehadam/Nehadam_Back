# 1. JDK 17 베이스 이미지 사용
FROM bellsoft/liberica-openjdk-alpine:17

CMD ["./gradlew", "clean", "build"]
# 2. 작업 디렉토리 설정
WORKDIR /app

ARG JAR_FILE=build/libs/*SNAPSHOT.jar
# 3. 빌드된 JAR 파일을 컨테이너로 복사
COPY ${JAR_FILE} app.jar

EXPOSE 8080
# 4. JAR 파일 실행
ENTRYPOINT ["java", "-jar", "/app/app.jar"]
