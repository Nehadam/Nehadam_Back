package com.example.nehadam.entity;

import com.fasterxml.jackson.annotation.JsonIgnore;
import jakarta.persistence.*;
import jakarta.persistence.criteria.CriteriaBuilder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Entity
@Getter
@Setter
@NoArgsConstructor
public class Image {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(unique = true)
    private String s3filename;

    private String originalFilename;

    private String filetype;

    private String fileurl;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "user_id")
    @JsonIgnore
    private User user;

    public Image(String s3filename, String originalFilename, String filetype, String fileurl, User user) {
        this.s3filename = s3filename;
        this.originalFilename = originalFilename;
        this.filetype = filetype;
        this.fileurl = fileurl;
        this.user = user;
    }
}
