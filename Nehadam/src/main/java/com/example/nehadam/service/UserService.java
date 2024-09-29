package com.example.nehadam.service;

import com.example.nehadam.dto.UserDTO;
import com.example.nehadam.entity.User;
import com.example.nehadam.exception.UserNotFoundException;
import com.example.nehadam.repository.UserRepository;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.Random;


@Service
public class UserService {

    private final UserRepository userRepository;

    public UserService(UserRepository userRepository) {
        this.userRepository = userRepository;
    }

    public void saveUserInfo(String email){
        if (userRepository.existsByEmail(email)){
            throw new IllegalArgumentException("Email already exists");
        }

        UserDTO userDTO = new UserDTO();
        userDTO.setEmail(email);
        userDTO.setUsername("Guest" + new Random().nextInt(99999));

        User user = new User(userDTO.getEmail(), userDTO.getUsername());
        userRepository.save(user);
    }

    public UserDTO getUserInfo(String email){
        User user = userRepository.findByEmail(email).orElseThrow(
                () -> new UserNotFoundException("User not found with email: " + email));

        UserDTO userDTO = new UserDTO();
        userDTO.setEmail(user.getEmail());
        userDTO.setUsername(user.getUsername());

        return userDTO;
    }

    public void updateUserInfo(UserDTO userDTO){
        User user = userRepository.findByEmail(userDTO.getEmail()).orElseThrow(
                () -> new UserNotFoundException("User not found with email: " + userDTO.getEmail()));

        user.updateUsername(userDTO.getUsername());
        userRepository.save(user);
    }

}
