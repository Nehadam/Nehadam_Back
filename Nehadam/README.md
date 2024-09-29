## 프로젝트 설정 가이드

1. **`application.properties` 파일 생성**:
    ```properties
    spring.application.name=Nehadam
    spring.profiles.include=db, s3  # application-db, application-s3 참조
    ```

---

2. **`application-db.properties` 파일 생성**:

    ### MySQL Database 연결
    ```properties
    spring.datasource.driver-class-name=com.mysql.cj.jdbc.Driver
    spring.datasource.url=jdbc:mysql://localhost/your_database_name?useSSL=false&useUnicode=true&serverTimezone=Asia/Seoul&allowPublicKeyRetrieval=true
    spring.datasource.username=your_username
    spring.datasource.password=your_password
    ```

    ### Available options: validate | update | create | create-drop | none
    - 처음 생성 시: `create`
    - 그 후로는: `none`

    ```properties
    spring.jpa.hibernate.ddl-auto=none
    spring.jpa.hibernate.naming.physical-strategy=org.hibernate.boot.model.naming.PhysicalNamingStrategyStandardImpl
    ```

---

3. **`application-s3.properties` 파일 생성**:

    ### AWS S3 버킷 연결
    ```properties
    cloud.aws.credentials.accessKey=
    cloud.aws.credentials.secretKey=
    cloud.aws.region.static=
    cloud.aws.s3.bucket=
    ```
