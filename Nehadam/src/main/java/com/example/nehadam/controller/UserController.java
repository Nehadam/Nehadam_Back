package com.example.nehadam.controller;

import com.example.nehadam.dto.UserDTO;
import com.example.nehadam.service.UserService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Random;

@RestController
@RequestMapping("/api/user")
@Tag(name = "User Management System", description = "Operations pertaining to user in User Management System")
public class UserController {
    private final UserService userService;

    public UserController(UserService userService) {
        this.userService = userService;
    }

    @Operation(summary = "유저 정보 저장", description = "유저 정보를 데이터베이스에 저장합니다.")
    @PostMapping("/saveUserInfo")
    public ResponseEntity<String> saveUserInfo(@RequestParam("email") String email){

        userService.saveUserInfo(email);
        return ResponseEntity.status(HttpStatus.CREATED).body("유저 정보 저장 성공");
    }

    @Operation(summary = "유저 정보 얻기", description = "이메일로 유저 정보를 받아옵니다.")
    @GetMapping("/getUserInfo")
    public ResponseEntity<UserDTO> getUserInfo(@RequestParam("email") String email){
        UserDTO userDTO = userService.getUserInfo(email);
        return ResponseEntity.ok(userDTO);
    }

    @Operation(summary = "유저 정보 수정", description = "유저 정보를 수정합니다.")
    @PostMapping("/updateUserInfo")
    public ResponseEntity<String> updateUserInfo(@RequestBody UserDTO userDTO){

        userService.updateUserInfo(userDTO);
        return ResponseEntity.ok("유저 정보 수정 성공");
    }
}
