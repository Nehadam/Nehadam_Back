package com.example.nehadam.controller;

import com.amazonaws.services.s3.AmazonS3;
import com.example.nehadam.entity.Image;
import com.example.nehadam.service.ImageService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.List;

@Controller
@RequestMapping("/api/{email}/images")
@RequiredArgsConstructor
@Tag(name = "Image System", description = "Image management")
public class ImageController {

    private final ImageService imageService;

    @Operation(summary = "이미지 저장", description = "이미지 정보를 데이터베이스에 저장. 이미지 관리를 ")
    @PostMapping(value = "/upload", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<String> upload(@RequestParam("file")MultipartFile file, @PathVariable String email) throws IOException {

        Image imageFile = imageService.saveFile(file, email);

        if (imageFile != null) {
            return ResponseEntity.ok("이미지 저장 성공");
        } else {
            return ResponseEntity.status(500).body("Failed to save image");
        }

    }

    @Operation(summary = "이미지 리스트 반환", description = "해당 유저의 이미지 리스트를 반환합니다.")
    @GetMapping("/list")
    public ResponseEntity<List<Image>> getUserImages(@PathVariable String email){
        List<Image> images = imageService.getUserImages(email);
        return ResponseEntity.ok(images);
    }

    @Operation(summary = "이미지 삭제", description = "해당 이미지를 삭제합니다.")
    @PostMapping("/delete")
    public ResponseEntity<String> deleteImage(@RequestParam("imageId") Long id){
        imageService.deleteImage(id);
        return ResponseEntity.ok("Deleted");
    }
}

