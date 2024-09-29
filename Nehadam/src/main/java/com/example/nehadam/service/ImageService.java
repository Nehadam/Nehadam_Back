package com.example.nehadam.service;

import com.example.nehadam.entity.Image;
import com.example.nehadam.entity.User;
import com.example.nehadam.exception.ImageNotFoundException;
import com.example.nehadam.exception.UserNotFoundException;
import com.example.nehadam.repository.ImageRepository;
import com.example.nehadam.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;


import java.io.IOException;
import java.util.List;


@Service
@RequiredArgsConstructor
public class ImageService {

    private final ImageRepository imageRepository;
    private final UserRepository userRepository;
    private final S3Service s3Service;

    public Image saveFile(MultipartFile file, String email) throws IOException{
        User user = userRepository.findByEmail(email)
                .orElseThrow(() -> new UserNotFoundException("User not found"));

        String fileUrl = s3Service.uploadFile(file);
        // String fileUrl = "https://s3.amazonaws.com/your-bucket-name/fake-file-name.jpg"; // s3없이 테스트할때 사용한 코드
        String s3FileName = fileUrl.substring(fileUrl.lastIndexOf('/') + 1);

        Image image = new Image(file.getOriginalFilename(), s3FileName, file.getContentType(), fileUrl, user);

        user.getImages().add(image);
        userRepository.save(user);

        return image;
    }

    public List<Image> getUserImages(String email){
        User user = userRepository.findByEmail(email)
                .orElseThrow(() -> new UserNotFoundException("User not found"));

        return user.getImages();
    }

    public void deleteImage(Long imageId){
        Image image = imageRepository.findById(imageId)
                .orElseThrow(() -> new ImageNotFoundException("Image not found"));

        s3Service.deleteFile(image.getS3filename()); // s3없이 테스트해보려면 주석처리
        imageRepository.delete(image);
    }



}
