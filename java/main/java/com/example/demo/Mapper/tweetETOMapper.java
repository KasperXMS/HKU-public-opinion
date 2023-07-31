package com.example.demo.Mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.example.demo.Entity.tweetDto;
import com.example.demo.Entity.tweetETO;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.util.List;

@Mapper
public interface tweetETOMapper extends BaseMapper<tweetETO> {
    @Select("${findtext}")
    List<tweetETO> FindText(@Param("findtext")String sql);
}
